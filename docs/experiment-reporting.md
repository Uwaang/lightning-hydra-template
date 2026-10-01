# Experiment reporting

When a Lightning logger is configured, the default training extras enable a
research-oriented classification report after the best checkpoint has been evaluated.
Fast-dev runs skip the reporting pass.

For classification models whose `predict_step` returns `logits`, `preds`, and
`targets`, the test report contains:

- scalar summary metrics: accuracy, macro precision/recall/F1, and weighted F1;
- `reports/test/confusion_matrix.png` and the raw CSV matrix;
- per-class precision, recall, F1, and support as JSON and CSV;
- one row per test sample with target, prediction, confidence, correctness, and top-k
  probabilities;
- an MLflow table for direct sample-level inspection when MLflow is the active logger.

The same MLflow run also receives the best Lightning checkpoint, best/current plain
PyTorch state dicts, Hydra configuration, train log, package/runtime/Git/GPU metadata,
and source/config snapshots. Existing `artifacts/` or `exports/` directories under
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
  log_existing_exports: true
```

Disable expensive or large artifacts explicitly for large experiments. For example,
`extras.reporting.log_prediction_table=false` keeps the report CSV while avoiding the
additional MLflow table.

Local MLflow tracking uses the SQLite database and artifact root configured in
`configs/logger/mlflow.yaml`. Start the UI against that same database, for example:

```bash
python -m mlflow ui --backend-store-uri sqlite:///logs/mlflow/mlflow.db --port 5000
```

The reporting code produces ordinary files first and only the publishing adapter is
MLflow-specific. This keeps the report reusable by other tracking backends.
