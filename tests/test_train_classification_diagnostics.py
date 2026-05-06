"""Tests for classifier diagnostics artifacts and experiment summary plotting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import mmwave_radar_devtool.ml.train_classification as train_cls
from mmwave_radar_devtool.ml.data import LabeledCapture, SplitCaptures
from mmwave_radar_devtool.ml.plot_experiments import main as plot_experiments_main


@pytest.fixture()
def _fake_captures() -> list[LabeledCapture]:
    labels = ["a", "b", "c"]
    captures: list[LabeledCapture] = []
    for label in labels:
        for idx in range(4):
            captures.append(LabeledCapture(path=Path(f"{label}_{idx}.bin"), label=label))
    return captures


def test_train_classifier_cli_generates_diagnostics_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _fake_captures: list[LabeledCapture],
) -> None:
    """Training command should still run and emit diagnostic plot/csv artifacts."""

    def _fake_discover(*args: object, **kwargs: object) -> list[LabeledCapture]:
        return list(_fake_captures)

    def _fake_split(
        captures: list[LabeledCapture],
        *,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> SplitCaptures:
        by_label: dict[str, list[LabeledCapture]] = {}
        for capture in captures:
            by_label.setdefault(capture.label, []).append(capture)
        train: list[LabeledCapture] = []
        val: list[LabeledCapture] = []
        test: list[LabeledCapture] = []
        for label in sorted(by_label):
            items = sorted(by_label[label], key=lambda item: str(item.path))
            train.extend(items[:2])
            val.append(items[2])
            test.append(items[3])
        return SplitCaptures(train=tuple(train), val=tuple(val), test=tuple(test))

    def _fake_stack_split(
        captures: tuple[LabeledCapture, ...],
        *,
        class_to_index: dict[str, int],
        cfg_path: Path,
        range_side: str,
        window_kind: str,
        eps: float,
        max_frames_per_capture: int | None,
        baseline_open_mean_db: np.ndarray | None,
        baseline_blocked_mean_db: np.ndarray | None,
        range_min_m: float | None,
        range_max_m: float | None,
        sample_mode: str,
        feature_mode: str,
        target_range_m: float,
        range_gate_bins: int,
        background_subtraction: str,
        background_reference: object | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_rows: list[np.ndarray] = []
        y_rows: list[int] = []
        for capture in captures:
            class_idx = class_to_index[capture.label]
            for frame_idx in range(6):
                seed = sum(ord(ch) for ch in str(capture.path)) + frame_idx
                rng = np.random.default_rng(seed)
                vec = rng.normal(loc=0.0, scale=0.03, size=(12,)).astype(np.float32)
                vec[class_idx] += 1.5
                vec[class_idx + 3] += 0.7
                x_rows.append(vec)
                y_rows.append(class_idx)
        return np.stack(x_rows, axis=0), np.array(y_rows, dtype=np.int64)

    monkeypatch.setattr(train_cls, "discover_labeled_captures", _fake_discover)
    monkeypatch.setattr(train_cls, "split_captures_stratified", _fake_split)
    monkeypatch.setattr(train_cls, "_stack_split", _fake_stack_split)

    out_path = tmp_path / "water_classifier.pt"
    exit_code = train_cls.main(
        [
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--cfg",
            "config/xwr18xx_profile_raw_capture.cfg",
            "--out",
            str(out_path),
            "--epochs",
            "4",
            "--min-epochs",
            "1",
            "--early-stopping-patience",
            "2",
            "--batch-size",
            "64",
            "--plots",
            "--save-best-val",
        ]
    )
    assert exit_code == 0

    summary_path = out_path.with_suffix(".json")
    assert out_path.exists()
    assert summary_path.exists()

    stem = out_path.stem
    assert (tmp_path / f"{stem}_confusion_matrix.csv").exists()
    assert (tmp_path / f"{stem}_classification_report.csv").exists()
    assert (tmp_path / f"{stem}_pca_features.png").exists()
    assert (tmp_path / f"{stem}_pca_model_embedding.png").exists()
    assert (tmp_path / f"{stem}_training_curves.png").exists()
    assert (tmp_path / f"{stem}_confidence_histogram.png").exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    assert "best_val_acc" in metrics
    assert "test_acc" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics


def test_plot_experiments_builds_summary_and_bar_plots(tmp_path: Path) -> None:
    """Experiment plotter should build CSV + accuracy/F1 comparison figures."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    run_a = {
        "run_name": "run_a",
        "checkpoint": "/tmp/run_a.pt",
        "feature": {
            "feature_mode": "complex_coherent",
            "target_range_m": 0.4,
            "range_gate_bins": 4,
            "background_subtraction": "complex_range",
            "normalization_mode": "trainset_standardize",
        },
        "training": {"seed": 0},
        "metrics": {
            "best_val_acc": 0.7,
            "best_val_loss": 0.9,
            "test_acc": 0.65,
            "test_loss": 1.1,
            "macro_f1": 0.61,
            "weighted_f1": 0.63,
        },
    }
    run_b = {
        "run_name": "run_b",
        "checkpoint": "/tmp/run_b.pt",
        "feature": {
            "feature_mode": "zero_doppler_db",
            "target_range_m": 0.4,
            "range_gate_bins": 2,
            "background_subtraction": "none",
            "normalization_mode": "none",
        },
        "training": {"seed": 1},
        "metrics": {
            "best_val_acc": 0.75,
            "best_val_loss": 0.8,
            "test_acc": 0.72,
            "test_loss": 0.95,
            "macro_f1": 0.7,
            "weighted_f1": 0.71,
        },
    }

    (results_dir / "run_a.json").write_text(json.dumps(run_a), encoding="utf-8")
    (results_dir / "run_b.json").write_text(json.dumps(run_b), encoding="utf-8")

    out_csv = tmp_path / "experiment_summary.csv"
    exit_code = plot_experiments_main(
        [
            "--results-dir",
            str(results_dir),
            "--out",
            str(out_csv),
        ]
    )
    assert exit_code == 0
    assert out_csv.exists()
    assert (tmp_path / "experiment_summary_accuracy.png").exists()
    assert (tmp_path / "experiment_summary_f1.png").exists()
