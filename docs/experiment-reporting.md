# Experiment reporting

When a Lightning logger is configured, the default training extras enable a
research-oriented classification report after the best checkpoint has been evaluated.
Fast-dev runs skip the reporting pass.

For classification models whose `predict_step` returns `logits`, `preds`, and
`targets`, the test report contains:

- scalar summary metrics: accuracy, macro precision/recall/F1, and weighted F1;
- raw and true-class-normalized confusion matrices as PNG and CSV;
- per-class precision, recall, F1, and support as JSON and CSV;
- one row per test sample with target, prediction, confidence, correctness, and top-k
  probabilities;
- a bounded MLflow error-analysis table: misclassified samples first, then the
  lowest-confidence remaining samples;
- bounded qualitative image artifacts when image logging is enabled: one transformed
  train mini-batch, the same fixed validation samples at the first/final checkpoints,
  and a highest-confidence test-error gallery.

The same MLflow run also receives the best Lightning checkpoint, best/current plain
PyTorch state dicts, Hydra configuration, train log, package/runtime/Git/GPU metadata,
dataset provenance, and source/config snapshots. Standard scalar tracking also includes
epoch-level learning rate, train/validation duration, and lightweight CPU/GPU memory and
utilization metrics when the platform exposes them. Existing `artifacts/` or
`exports/` directories under
the run output are uploaded as export artifacts. PT2 or ONNX export is not forced during
training; export remains a separate capability because not every model/backend has the
same deployment contract.

The behavior is controlled in `configs/extras/default.yaml`:

```yaml
reporting:
  enabled: true
  top_k: 3
  class_names: null
  log_metadata: true
  log_state_dicts: true
  log_best_checkpoint: true
  log_prediction_table: true
  prediction_table_max_rows: 500
  log_learning_rate: true
  log_epoch_time: true
  log_system_metrics: true
  log_dataset_provenance: true
  log_existing_exports: true
  image_logging:
    enabled: true
    train_preview_images: 16
    fixed_val_images: 16
    confident_error_images: 32
    mean: null
    std: null
```

Disable expensive or large artifacts explicitly for large experiments. The full
prediction CSV remains the source of truth; `prediction_table_max_rows` only limits the
interactive MLflow table. Setting `extras.reporting.log_prediction_table=false` keeps
the full CSV while avoiding the additional table.

Dataset provenance uses a SHA-256 fingerprint over dataset identity/configuration and
split policy. Machine-local paths and dataloader execution settings are excluded. The
default fingerprint deliberately does not hash every raw sample byte; that scope is
recorded in `metadata/dataset.json`, and project-specific datamodules can provide a
more precise `dataset_provenance()` payload.

Local MLflow tracking uses the SQLite database and artifact root configured in
`configs/logger/mlflow.yaml`. Start the UI against that same database, for example:

```bash
python -m mlflow ui --backend-store-uri sqlite:///logs/mlflow/mlflow.db --port 5000
```

The reporting code produces ordinary files first and only the publishing adapter is
MLflow-specific. This keeps the report reusable by other tracking backends.
