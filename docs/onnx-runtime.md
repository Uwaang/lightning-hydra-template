# ONNX Runtime deployment

This project treats ONNX + ONNX Runtime (ORT) as the primary portable deployment path.
PT2 export remains available as a PyTorch-native graph artifact, but it is not required
for ONNX Runtime deployment.

## Supported execution-provider matrix

The benchmark API uses short provider names and maps them to explicit ORT stacks:

| name       | primary ORT EP              | fallback stack | typical target     |
| ---------- | --------------------------- | -------------- | ------------------ |
| `cpu`      | `CPUExecutionProvider`      | CPU only       | x86-64 / Arm64 CPU |
| `xnnpack`  | `XnnpackExecutionProvider`  | CPU            | x86 / Arm CPU      |
| `cuda`     | `CUDAExecutionProvider`     | CPU            | NVIDIA GPU         |
| `tensorrt` | `TensorrtExecutionProvider` | CUDA, CPU      | NVIDIA GPU         |

`benchmark_onnx()` rejects a requested provider when it is not compiled into the
installed ORT package. It also verifies that the primary provider is still active after
session creation, so provider-load/session fallback cannot be mislabeled as CUDA or
TensorRT. ORT may still assign unsupported nodes to lower-priority EPs within an active
provider stack.

## Environment recipes

The locked `onnx` extra is the CPU-oriented development environment:

```bash
uv sync --extra onnx
```

For NVIDIA deployment, use an environment that contains `onnxruntime-gpu` instead of
the CPU `onnxruntime` distribution. Keep this separate from the locked CPU environment
to avoid two distributions providing the same `onnxruntime` Python package.

TensorRT EP additionally needs a TensorRT runtime that matches the ORT GPU build.
The Windows validation for this repository used ORT 1.30.0, CUDA 13.0, cuDNN 9,
and TensorRT 10.16.1.11.

The standard Python wheel used by the CPU development environment does not expose
XNNPACK. For Windows or Linux, build ONNX Runtime from source with `--use_xnnpack`
(and `--build_wheel` when a Python wheel is wanted).

## Runtime benchmark semantics

Runtime setup and export time are excluded. Each result reports mean, median, p90, p95,
minimum and maximum latency, median-based throughput, and the raw repeated samples.

Inputs passed to the benchmark API are CPU tensors. For CUDA and TensorRT this means
the measured `session.run()` latency includes the normal host-to-device and
device-to-host path. This is intentional: it is an end-to-end host-call measurement,
not a kernel-only GPU benchmark.

Run a provider matrix for one model:

```bash
python scripts/ort_provider_matrix.py artifacts/model.onnx \
  --input-shape 1,3,224,224 \
  --providers cpu xnnpack cuda tensorrt \
  --num-threads 1 \
  --output artifacts/provider-matrix.json
```

Unavailable providers are recorded as unavailable instead of silently falling back.

## Static INT8 post-training quantization

`src.utils.quantize_onnx_static` implements the current small PTQ path:

- ONNX Runtime quantization pre-processing;
- static calibration;
- QDQ representation;
- signed INT8 activations and weights;
- per-channel weight quantization by default;
- MinMax calibration by default.

The CIFAR-10 validation helper reuses a trained ResNet-18 state dict, exports FP32 ONNX,
calibrates on a bounded validation subset, creates INT8 ONNX, then compares accuracy,
model size and CPU latency:

```bash
python scripts/cifar10_ort_spike.py \
  --state-dict logs/cifar10-gpu-10ep/best_state_dict.pt \
  --output-dir artifacts/cifar10-ort-spike
```

Quantization is an optimization candidate, not an assumed speedup. Always benchmark the
target CPU and execution provider.

## Reference validation

A local validation run used the repository's 10-epoch CIFAR-10 ResNet-18 checkpoint.
These numbers are hardware-specific examples, not project performance guarantees.

Environment: 2026-10-02, Windows 11, Ryzen 5 5600, GeForce GTX 1660,
ORT GPU 1.30.0 and TensorRT 10.16.1.11. Session creation is excluded and CPU-host
transfer is included.
The provider matrix below used six ORT intra-op threads.

| batch | CPU EP median | CUDA EP median | TensorRT EP median |
| ----: | ------------: | -------------: | -----------------: |
|     1 |      0.396 ms |       2.439 ms |           1.025 ms |
|    32 |      7.433 ms |       3.192 ms |           2.008 ms |
|   128 |     28.152 ms |       5.405 ms |           3.365 ms |

For this small 32x32 workload, CPU wins at batch 1 while CUDA/TensorRT overtake it
at larger batches. CPU/CUDA/TensorRT FP32 outputs agreed within about `1.4e-6`
maximum absolute error.

Static PTQ produced:

| artifact  | test accuracy |     size |
| --------- | ------------: | -------: |
| FP32 ONNX |        77.49% | 44.81 MB |
| INT8 QDQ  |        76.96% | 11.36 MB |

The INT8 model was about 25.3% of the FP32 file size with a 0.53 percentage-point
accuracy drop. It did not improve latency on the tested Ryzen 5 5600 CPU EP across
the checked 1/6/12-thread settings, so INT8 is not promoted as the default path.

## x86 Windows / Linux recipe

1. Export ONNX and verify numerical parity.
2. Start with `CPUExecutionProvider`.
3. Sweep at least one thread and the machine's physical-core range; do not assume that
   more threads improve batch-1 latency.
4. Try static INT8 PTQ only after the FP32 baseline is recorded.
5. If the ORT build includes XNNPACK, run the same model and input through `xnnpack`.
6. On NVIDIA systems, add CUDA and then TensorRT only when the workload is large enough
   to justify accelerator and engine-build overhead.
7. Save the provider matrix JSON with the experiment artifacts.

The benchmark reports provider availability explicitly, which is useful when the same
recipe is run across developer workstations and deployment machines.

## Arm64 Linux recipe

1. Install or build an Arm64-native ONNX Runtime package.
2. Record a baseline with `CPUExecutionProvider`.
3. For XNNPACK, use an ORT build compiled with `--use_xnnpack`.
4. Re-run the same provider-matrix command with the same model, batch and thread settings.
5. Sweep the thread count against physical CPU cores and record power/thermal constraints
   separately when the target is an embedded board.
6. Run FP32 first, then INT8 PTQ as an A/B measurement.

No Arm64 performance number is committed yet because this repository has not been
validated on a physical Arm64 target. The recipe is intentionally hardware-neutral so
the same JSON benchmark can be collected when such a target is available.

## Deployment choice

Use the simplest runtime that satisfies the target:

- x86 CPU: ORT CPU EP first;
- Arm CPU: ORT CPU EP first, then XNNPACK as an A/B candidate;
- NVIDIA GPU: CUDA EP first, TensorRT EP when its extra build/runtime complexity pays off;
- INT8: keep it only when the target shows an acceptable accuracy/size/latency trade-off.

The project deliberately does not require ExecuTorch, PT2E or torchao for this ONNX
deployment path.

## Official references

- Execution Providers: https://onnxruntime.ai/docs/execution-providers/
- EP builds and XNNPACK: https://onnxruntime.ai/docs/build/eps.html
- Quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- CUDA EP: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- TensorRT EP: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
