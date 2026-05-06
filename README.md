# mmwave_radar_devtool

Utilities for recording raw TI mmWave radar captures and visualizing saved radar `.bin`
files.

This README covers only:

- `dataset_recorder.bash`
- the `mmw capture` single capture command
- the `mmw live` terminal viewer
- `radar_bin_visualizer_fixed.py`

## Installation

Install UV if you don't have it yet:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repo and install the Python dependencies with:

```bash
uv sync
```

The recorder uses the `mmw` command provided by this project, so run it from the
repository root after installing the environment.

## Single Capture

Use `mmw capture` to record one raw radar `.bin` file for a fixed duration:

```bash
uv run mmw capture \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --duration 5 \
  --output capture.bin
```

The command prints capture statistics after the recording finishes, including
received packets, bytes written, elapsed time, and detected sequence gaps.

Use `--keep-dca-header` if the DCA1000 packet headers should stay in the output
file instead of being stripped.

## Live Viewer

Use `mmw live` to run the terminal live viewer with the radar configuration:

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg
```

Run the live viewer for a fixed duration:

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --duration 10
```

Record while using the live viewer:

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --output live_capture.bin
```

Show live range-delta against a saved baseline capture:

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --live-baseline-capture baseline_open.bin
```

## Dataset Recorder

`dataset_recorder.bash` repeatedly records 5 second raw radar captures using:

- config file: `config/xwr18xx_profile_raw_capture.cfg`
- radar CLI serial port: `/dev/ttyACM0`
- output format: `<label>_<record_num>.bin`

Run it with:

```bash
bash dataset_recorder.bash
```

The script prompts for a recording label, then writes files such as:

```text
person_0.bin
person_1.bin
person_2.bin
```

After each capture, choose the next action:

- press `Enter` or type `c` to continue with the same label
- type `l` to enter a new label and restart numbering at `0`
- type `q` to quit

## Radar Bin Visualizer

`radar_bin_visualizer_fixed.py` now builds TI-demo-like range tensors for
training workflows from DCA1000 raw ADC captures.

Typical usage with object capture and optional empty/background capture:

```bash
uv run python radar_bin_visualizer_fixed.py object_capture.bin \
  --empty-capture empty_capture.bin \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --output-dir outputs \
  --window-frames 4
```

Dual-baseline attenuation comparison (no-attenuation + max-attenuation):

```bash
uv run python radar_bin_visualizer_fixed.py object_capture.bin \
  --baseline-open-capture baseline_open.bin \
  --baseline-blocked-capture baseline_blocked.bin \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --output-dir outputs_dual_baseline
```

Saved outputs include:

```text
range_profile_zero_doppler.npy
range_doppler_heatmap.npy
nn_logmag_windows.npy
target_bin_report.json
debug_range_profile_slice.png
debug_zero_doppler_profile_avg.png
debug_range_doppler_unshifted.png
debug_range_doppler_shifted.png
```

When baseline captures are provided, additional outputs are saved:

```text
delta_vs_open_baseline_logmag.npy
delta_vs_blocked_baseline_logmag.npy
debug_delta_baseline_profiles.png
delta_baseline_summary.json
```

Useful options:

- `--window-kind hann|rect`
- `--window-frames N`
- `--window-step N`
- `--target-m 0.4`
- `--useful-side full|positive|negative`

## ML Training and Visualization

The project now includes a dedicated ML package under `mmwave_radar_devtool.ml`
for:

- NN classification training from `.bin` captures
- NN regression training from `.bin` captures
- per-capture and dataset-level visualization of average range-Doppler and
  range profiles per receiver

For nested datasets like `dataset_cls/<label>/*.bin`, the dataset visualizer can
also stack all runs from each class on top of each other after the range FFT:

```bash
uv run mmw-visualize-dataset \
  --dataset-dir ./dataset_cls \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --group-by-label \
  --plot-run-overlays \
  --output-dir outputs/ml_visuals
```

This saves the usual per-label averages plus an extra overlay plot for each
class, which makes it easier to compare how the range profile changes with
distance across the repeated runs.

Install ML dependencies:

```bash
uv sync --extra ml
```

### Train a Classifier

Works with either:

- nested layout: `dataset/<label>/*.bin`
- flat layout from `dataset_recorder.bash`: `<label>_<index>.bin`

```bash
uv run mmw-train-classifier \
  --dataset-dir ./dataset \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --plots \
  --save-best-val \
  --early-stopping-patience 8 \
  --out outputs/ml/radar_classifier.pt
```

Train directly on baseline-delta features (if baseline flags are passed):

```bash
uv run mmw-train-classifier \
  --dataset-dir ./dataset \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --baseline-open-capture baseline_open.bin \
  --baseline-blocked-capture baseline_blocked.bin \
  --sample-mode capture-mean-std \
  --range-min-m 0.25 \
  --range-max-m 0.70 \
  --out outputs/ml/radar_classifier_delta.pt
```

For static-target experiments, `--sample-mode capture-mean` or
`--sample-mode capture-mean-std` is usually better than treating every frame
inside one capture as an independent sample.

You can also train on complex coherent features instead of zero-Doppler dB:

```bash
uv run mmw-train-classifier \
  --dataset-dir ./dataset \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --feature-mode complex_coherent \
  --target-range-m 0.40 \
  --range-gate-bins 2 \
  --background-subtraction complex_range \
  --background-capture empty_scene.bin \
  --sample-mode capture-mean-std \
  --normalization-mode trainset_standardize \
  --plots \
  --out outputs/ml/radar_classifier_complex.pt
```

When `--plots` is enabled, training writes:

- `<run>_training_curves.png`
- `<run>_confusion_matrix.png`
- `<run>_confusion_matrix_normalized.png`
- `<run>_confusion_matrix.csv`
- `<run>_classification_report.csv`
- `<run>_pca_features.png`
- `<run>_pca_model_embedding.png`
- `<run>_confidence_histogram.png`

and includes `best_val_acc`, `best_val_loss`, `test_acc`, `test_loss`,
`macro_f1`, and `weighted_f1` in the JSON summary.

### Train a Regressor

Targets can come from:

- `--label-target-map` JSON (`{\"label\": value}`)
- or parsed numeric token from class/file names (default regex)

```bash
uv run mmw-train-regressor \
  --dataset-dir ./dataset \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --label-target-map targets.json \
  --out outputs/ml/radar_regressor.pt
```

Baseline-delta regression works the same way:

```bash
uv run mmw-train-regressor \
  --dataset-dir ./dataset \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --baseline-open-capture baseline_open.bin \
  --baseline-blocked-capture baseline_blocked.bin \
  --sample-mode capture-mean-std \
  --range-min-m 0.25 \
  --range-max-m 0.70 \
  --label-target-map targets.json \
  --out outputs/ml/radar_regressor_delta.pt
```

Complex coherent regression uses the same feature flags:
`--feature-mode complex_coherent --target-range-m ... --range-gate-bins ...`.

### Evaluate an Existing Classifier Checkpoint

Regenerate confusion matrix, report, PCA plots, and confidence histogram:

```bash
uv run mmw-evaluate-classifier \
  --checkpoint outputs/ml/radar_classifier_complex.pt \
  --dataset-dir ./dataset \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --out-dir outputs/ml/eval_radar_classifier_complex \
  --split test \
  --plots
```

### Evaluate Regressor Predictions (MAE / RMSE / R2)

First, run prediction on a holdout set:

```bash
uv run mmw-predict \
  --checkpoint outputs/ml/water_regressor_timeholdout.pt \
  --json \
  dataset_open_holdout_last7/*/*.bin \
  > outputs/ml/eval_timeholdout_last7/water_regressor_timeholdout_preds.json
```

Then evaluate predictions against your label map (includes "all" and
"without none" metrics):

```bash
uv run mmw-evaluate-regressor-preds \
  --predictions-json outputs/ml/eval_timeholdout_last7/water_regressor_timeholdout_preds.json \
  --label-target-map targets_open.json \
  --none-label none \
  --plots \
  --out-json outputs/ml/eval_timeholdout_last7/water_regressor_timeholdout_eval.json
```

### Run a Grid of Classifier Experiments

Use JSON (or YAML if `PyYAML` is installed) to define sweeps:

```bash
uv run mmw-run-experiments \
  --config experiments/static_water_grid.json \
  --plots
```

`mmw-run-experiments` skips unsupported combinations and existing completed
runs unless `--force` is passed.

### Aggregate Experiment Results

Build a summary CSV and comparison charts across all run summaries:

```bash
uv run mmw-plot-experiments \
  --results-dir outputs/ml/grid_static_water \
  --out outputs/ml/grid_static_water/experiment_summary.csv
```

### Predict on New Captures

Run a trained classifier or regressor on one or more recorded `.bin` files:

```bash
uv run mmw-predict \
  --checkpoint outputs/ml/radar_classifier_delta.pt \
  new_capture.bin
```

For live NN predictions, pass the checkpoint to the live viewer:

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --live-ml-checkpoint outputs/ml/radar_classifier_delta.pt \
  --live-ml-window-frames 16
```

Live NN mode currently supports checkpoints trained with
`--feature-mode zero_doppler_db`.

### Visualize One Capture (`.bin`)

This produces:

- average range-Doppler heatmaps by receiver
- average range profile curves by receiver
- corresponding `.npy` arrays

```bash
uv run mmw-visualize-bin \
  reading.bin \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --output-dir outputs/ml_visuals
```

### Visualize All Captures in a Dataset

Aggregate across all `.bin` files in a directory:

```bash
uv run mmw-visualize-dataset \
  --dataset-dir ./dataset \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --output-dir outputs/ml_visuals
```

Optional per-label aggregation:

```bash
uv run mmw-visualize-dataset \
  --dataset-dir ./dataset \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --group-by-label
```

## License

This project is licensed under the Apache License 2.0. See `LICENSE`.
