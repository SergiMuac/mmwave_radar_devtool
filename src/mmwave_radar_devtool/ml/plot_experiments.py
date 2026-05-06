"""Summarize experiment JSON files into tables and comparison plots."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _to_float(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _abbreviate_label(label: str, *, max_len: int = 42) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def _extract_row(path: Path, payload: dict[str, Any]) -> dict[str, object]:
    feature = dict(payload.get("feature") or {})
    training = dict(payload.get("training") or {})
    metrics = dict(payload.get("metrics") or {})

    run_name = str(payload.get("run_name") or path.stem)
    checkpoint = payload.get("checkpoint")

    row: dict[str, object] = {
        "run_name": run_name,
        "feature_mode": feature.get("feature_mode"),
        "target_range_m": feature.get("target_range_m"),
        "range_gate_bins": feature.get("range_gate_bins"),
        "background_subtraction": feature.get("background_subtraction"),
        "normalization_mode": feature.get("normalization_mode"),
        "seed": training.get("seed"),
        "best_val_acc": _to_float(metrics.get("best_val_acc")),
        "best_val_loss": _to_float(metrics.get("best_val_loss")),
        "test_acc": _to_float(metrics.get("test_acc") or metrics.get("accuracy")),
        "test_loss": _to_float(metrics.get("test_loss") or metrics.get("loss")),
        "macro_f1": _to_float(metrics.get("macro_f1")),
        "weighted_f1": _to_float(metrics.get("weighted_f1")),
        "checkpoint_path": checkpoint,
        "summary_path": str(path.resolve()),
    }
    return row


def _plot_bar(
    rows: list[dict[str, object]],
    *,
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    filtered = [row for row in rows if _to_float(row.get(value_key)) is not None]
    if not filtered:
        return

    labels = [str(row["run_name"]) for row in filtered]
    values = np.array([float(row[value_key]) for row in filtered], dtype=np.float32)
    order = np.argsort(values)[::-1]

    labels_sorted = [labels[idx] for idx in order]
    values_sorted = values[order]
    display_labels = [_abbreviate_label(label) for label in labels_sorted]
    item_count = len(values_sorted)

    width = max(8.0, item_count * 0.48)
    height = 6.6 if item_count > 14 else 5.2
    fig, ax = plt.subplots(figsize=(width, height))
    bars = ax.bar(np.arange(len(values_sorted)), values_sorted, color="#2563eb")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(values_sorted)))
    ax.set_xticklabels(
        display_labels,
        rotation=60,
        ha="right",
        fontsize=7 if item_count > 20 else 8,
    )
    ax.grid(True, axis="y", alpha=0.2)

    for idx, bar in enumerate(bars):
        value = float(values_sorted[idx])
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    y_max = float(np.max(values_sorted)) if item_count else 1.0
    ax.set_ylim(0.0, max(1.0, y_max * 1.14 + 0.02))
    bottom_margin = 0.50 if item_count > 30 else 0.42 if item_count > 14 else 0.30
    fig.subplots_adjust(left=0.08, right=0.995, top=0.90, bottom=bottom_margin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse experiment-summary plotting arguments."""
    parser = argparse.ArgumentParser(description="Aggregate experiment JSON summaries.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output summary CSV path, e.g. outputs/ml/experiment_summary.csv",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build experiment CSV summary and comparison bar plots."""
    args = parse_args(argv)

    json_paths = sorted(path for path in args.results_dir.rglob("*.json") if path.is_file())
    if not json_paths:
        raise RuntimeError(f"No JSON summaries found under {args.results_dir}")

    rows: list[dict[str, object]] = []
    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if "metrics" not in payload:
            continue
        rows.append(_extract_row(path, payload))

    if not rows:
        raise RuntimeError(f"No compatible summary JSON files found under {args.results_dir}")

    columns = [
        "run_name",
        "feature_mode",
        "target_range_m",
        "range_gate_bins",
        "background_subtraction",
        "normalization_mode",
        "seed",
        "best_val_acc",
        "best_val_loss",
        "test_acc",
        "test_loss",
        "macro_f1",
        "weighted_f1",
        "checkpoint_path",
        "summary_path",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    accuracy_plot = args.out.parent / "experiment_summary_accuracy.png"
    f1_plot = args.out.parent / "experiment_summary_f1.png"
    _plot_bar(
        rows,
        value_key="test_acc",
        title="Test Accuracy Across Runs",
        ylabel="Accuracy",
        output_path=accuracy_plot,
    )
    _plot_bar(
        rows,
        value_key="macro_f1",
        title="Macro F1 Across Runs",
        ylabel="Macro F1",
        output_path=f1_plot,
    )

    print(f"Saved summary CSV: {args.out}")
    print(f"Saved accuracy plot: {accuracy_plot}")
    print(f"Saved macro-F1 plot: {f1_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
