"""Evaluate trained classifier checkpoints and generate diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..range_profile_dataset import parse_tensor_cfg
from .classification_eval import (
    pca_project_2d,
    plot_confidence_histogram,
    plot_confusion_matrix,
    plot_pca_scatter,
    save_classification_report_csv,
    save_confusion_matrix_csv,
    summarize_multiclass_predictions,
)
from .data import (
    BACKGROUND_SUBTRACTION_NONE,
    FEATURE_MODE_COMPLEX_COHERENT,
    FEATURE_MODE_ZERO_DOPPLER_DB,
    NORMALIZATION_NONE,
    NORMALIZATION_PER_RECORDING,
    NORMALIZATION_TRAINSET,
    BackgroundReference,
    LabeledCapture,
    discover_labeled_captures,
    extract_frame_feature_tensor_by_rx_db,
    feature_tensor_to_samples,
    load_background_reference,
    load_baseline_zero_doppler_mean_by_rx_db,
    per_recording_standardize_features,
    split_captures_stratified,
)
from .models import RadarMLP


def _optional_float(raw: object) -> float | None:
    if raw is None:
        return None
    value = float(raw)
    return value


def _path_or_none(raw: object) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return Path(text)


def _stack_split(
    captures: tuple[LabeledCapture, ...],
    *,
    class_to_index: dict[str, int],
    cfg_path: Path,
    feature: dict[str, Any],
    baseline_open_mean_db: np.ndarray | None,
    baseline_blocked_mean_db: np.ndarray | None,
    background_reference: BackgroundReference | None,
    max_frames_per_capture: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = parse_tensor_cfg(cfg_path)
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for capture in captures:
        frame_tensor = extract_frame_feature_tensor_by_rx_db(
            capture.path,
            cfg,
            range_side=str(feature.get("range_side", "positive")),
            window_kind=str(feature.get("window_kind", "hann")),
            eps=float(feature.get("eps", 1e-9)),
            max_frames=max_frames_per_capture,
            baseline_open_mean_db=baseline_open_mean_db,
            baseline_blocked_mean_db=baseline_blocked_mean_db,
            range_min_m=_optional_float(feature.get("range_min_m")),
            range_max_m=_optional_float(feature.get("range_max_m")),
            feature_mode=str(feature.get("feature_mode", FEATURE_MODE_ZERO_DOPPLER_DB)),
            target_range_m=float(feature.get("target_range_m", 0.40)),
            range_gate_bins=int(feature.get("range_gate_bins", 2)),
            background_subtraction=str(
                feature.get("background_subtraction", BACKGROUND_SUBTRACTION_NONE)
            ),
            background_reference=background_reference,
        )
        if frame_tensor.shape[0] == 0:
            continue
        x = feature_tensor_to_samples(
            frame_tensor,
            sample_mode=str(feature.get("sample_mode", "frames")),
        )
        y = np.full((x.shape[0],), class_to_index[capture.label], dtype=np.int64)
        all_x.append(x)
        all_y.append(y)

    if not all_x:
        raise RuntimeError("No samples were produced for requested split.")

    return (
        np.concatenate(all_x, axis=0).astype(np.float32, copy=False),
        np.concatenate(all_y, axis=0).astype(np.int64, copy=False),
    )


@torch.no_grad()
def _collect_logits(
    model: RadarMLP,
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)))
    loader = DataLoader(ds, batch_size=max(1, int(batch_size)), shuffle=False)
    model.eval()
    out: list[np.ndarray] = []
    for (xb,) in loader:
        logits = model(xb.to(device))
        out.append(logits.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)


@torch.no_grad()
def _collect_embedding(
    model: RadarMLP,
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)))
    loader = DataLoader(ds, batch_size=max(1, int(batch_size)), shuffle=False)
    model.eval()
    out: list[np.ndarray] = []
    for (xb,) in loader:
        emb = model.forward_features(xb.to(device))
        out.append(emb.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse evaluation command arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained classifier checkpoint and regenerate diagnostics plots."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which split to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--max-frames-per-capture", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run classifier checkpoint evaluation and artifact generation."""
    args = parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    class_names_raw = checkpoint.get("class_names")
    if class_names_raw is None:
        raise RuntimeError("Checkpoint has no class_names; use a classification checkpoint.")
    class_names = [str(item) for item in class_names_raw]
    class_to_index = {label: idx for idx, label in enumerate(class_names)}

    feature = dict(checkpoint.get("feature") or {})
    training = dict(checkpoint.get("training") or {})
    cfg_path = args.cfg or _path_or_none(feature.get("cfg_path"))
    if cfg_path is None:
        raise RuntimeError("Missing cfg path; pass --cfg explicitly.")
    cfg = parse_tensor_cfg(cfg_path)

    split_seed = int(args.seed if args.seed is not None else training.get("seed", 42))
    split_val_ratio = float(
        args.val_ratio if args.val_ratio is not None else training.get("val_ratio", 0.15)
    )
    split_test_ratio = float(
        args.test_ratio if args.test_ratio is not None else training.get("test_ratio", 0.15)
    )

    flat_label_regex = str(training.get("flat_label_regex", r"^(?P<label>.+)_(?P<index>\d+)$"))
    captures = discover_labeled_captures(args.dataset_dir, flat_label_regex=flat_label_regex)
    split = split_captures_stratified(
        captures,
        val_ratio=split_val_ratio,
        test_ratio=split_test_ratio,
        seed=split_seed,
    )
    selected = {"train": split.train, "val": split.val, "test": split.test}[args.split]

    baseline_open_path = _path_or_none(feature.get("baseline_open_capture"))
    baseline_blocked_path = _path_or_none(feature.get("baseline_blocked_capture"))
    background_path = _path_or_none(feature.get("background_capture"))

    baseline_open_mean_db = None
    baseline_blocked_mean_db = None
    background_reference = None

    feature_mode = str(feature.get("feature_mode", FEATURE_MODE_ZERO_DOPPLER_DB))
    if feature_mode == FEATURE_MODE_ZERO_DOPPLER_DB:
        if baseline_open_path is not None:
            baseline_open_mean_db = load_baseline_zero_doppler_mean_by_rx_db(
                baseline_open_path,
                cfg,
                range_side=str(feature.get("range_side", "positive")),
                window_kind=str(feature.get("window_kind", "hann")),
                eps=float(feature.get("eps", 1e-9)),
                range_min_m=_optional_float(feature.get("range_min_m")),
                range_max_m=_optional_float(feature.get("range_max_m")),
            )
        if baseline_blocked_path is not None:
            baseline_blocked_mean_db = load_baseline_zero_doppler_mean_by_rx_db(
                baseline_blocked_path,
                cfg,
                range_side=str(feature.get("range_side", "positive")),
                window_kind=str(feature.get("window_kind", "hann")),
                eps=float(feature.get("eps", 1e-9)),
                range_min_m=_optional_float(feature.get("range_min_m")),
                range_max_m=_optional_float(feature.get("range_max_m")),
            )
    elif feature_mode == FEATURE_MODE_COMPLEX_COHERENT:
        background_reference = load_background_reference(
            background_path,
            cfg,
            window_kind=str(feature.get("window_kind", "hann")),
            eps=float(feature.get("eps", 1e-9)),
            background_subtraction=str(
                feature.get("background_subtraction", BACKGROUND_SUBTRACTION_NONE)
            ),
        )
    else:
        raise ValueError(f"Unsupported feature mode in checkpoint: {feature_mode!r}")

    max_frames = args.max_frames_per_capture
    if max_frames is None:
        raw_max = feature.get("max_frames_per_capture")
        max_frames = None if raw_max is None else int(raw_max)

    x, y_true = _stack_split(
        selected,
        class_to_index=class_to_index,
        cfg_path=cfg_path,
        feature=feature,
        baseline_open_mean_db=baseline_open_mean_db,
        baseline_blocked_mean_db=baseline_blocked_mean_db,
        background_reference=background_reference,
        max_frames_per_capture=max_frames,
    )

    normalization = dict(checkpoint.get("normalization") or {})
    mean = np.asarray(normalization.get("mean", np.zeros((x.shape[1],))), dtype=np.float32)
    std = np.asarray(normalization.get("std", np.ones((x.shape[1],))), dtype=np.float32)
    std = np.maximum(std, 1e-6)

    normalization_mode = str(feature.get("normalization_mode", NORMALIZATION_TRAINSET))
    if normalization_mode == NORMALIZATION_TRAINSET:
        x_model = ((x - mean[None, :]) / std[None, :]).astype(np.float32)
    elif normalization_mode == NORMALIZATION_PER_RECORDING:
        x_model = per_recording_standardize_features(x)
    elif normalization_mode == NORMALIZATION_NONE:
        x_model = np.asarray(x, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization_mode!r}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = RadarMLP(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=len(class_names),
        hidden_dims=tuple(int(v) for v in checkpoint["hidden_dims"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    logits = _collect_logits(model, x_model, batch_size=int(args.batch_size), device=device)
    cm, y_pred, confidence, correct_mask, report = summarize_multiclass_predictions(
        logits,
        y_true,
        class_names=class_names,
    )

    loss = torch.nn.CrossEntropyLoss()(torch.from_numpy(logits), torch.from_numpy(y_true)).item()
    accuracy = float(np.mean(y_pred == y_true)) if y_true.size else 0.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.checkpoint.stem

    confusion_csv = args.out_dir / f"{run_name}_confusion_matrix.csv"
    confusion_norm_csv = args.out_dir / f"{run_name}_confusion_matrix_normalized.csv"
    report_csv = args.out_dir / f"{run_name}_classification_report.csv"

    save_confusion_matrix_csv(
        cm,
        class_names=class_names,
        output_path=confusion_csv,
        normalized=False,
    )
    save_confusion_matrix_csv(
        cm,
        class_names=class_names,
        output_path=confusion_norm_csv,
        normalized=True,
    )
    save_classification_report_csv(report, output_path=report_csv)

    artifacts = {
        "confusion_matrix_csv": str(confusion_csv.resolve()),
        "confusion_matrix_normalized_csv": str(confusion_norm_csv.resolve()),
        "classification_report_csv": str(report_csv.resolve()),
    }

    if args.plots:
        confusion_png = args.out_dir / f"{run_name}_confusion_matrix.png"
        confusion_norm_png = args.out_dir / f"{run_name}_confusion_matrix_normalized.png"
        pca_features_png = args.out_dir / f"{run_name}_pca_features.png"
        pca_embed_png = args.out_dir / f"{run_name}_pca_model_embedding.png"
        confidence_png = args.out_dir / f"{run_name}_confidence_histogram.png"

        plot_confusion_matrix(
            cm,
            class_names=class_names,
            output_path=confusion_png,
            normalized=False,
        )
        plot_confusion_matrix(
            cm,
            class_names=class_names,
            output_path=confusion_norm_png,
            normalized=True,
        )
        coords_x, explained_x = pca_project_2d(x_model)
        plot_pca_scatter(
            coords_x,
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            explained_variance=explained_x,
            title=f"PCA of Input Features ({args.split} split)",
            output_path=pca_features_png,
        )

        embedding = _collect_embedding(
            model,
            x_model,
            batch_size=int(args.batch_size),
            device=device,
        )
        coords_e, explained_e = pca_project_2d(embedding)
        plot_pca_scatter(
            coords_e,
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            explained_variance=explained_e,
            title=f"PCA of Model Embedding ({args.split} split)",
            output_path=pca_embed_png,
        )

        plot_confidence_histogram(
            confidence,
            correct_mask=correct_mask,
            output_path=confidence_png,
        )

        artifacts.update(
            {
                "confusion_matrix": str(confusion_png.resolve()),
                "confusion_matrix_normalized": str(confusion_norm_png.resolve()),
                "pca_features": str(pca_features_png.resolve()),
                "pca_model_embedding": str(pca_embed_png.resolve()),
                "confidence_histogram": str(confidence_png.resolve()),
            }
        )

    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "macro_precision": float(report.macro_precision),
        "macro_recall": float(report.macro_recall),
        "macro_f1": float(report.macro_f1),
        "weighted_precision": float(report.weighted_precision),
        "weighted_recall": float(report.weighted_recall),
        "weighted_f1": float(report.weighted_f1),
        "sample_count": int(y_true.size),
        "capture_count": len(selected),
        "classification_report": {
            "rows": list(report.rows),
            "accuracy": float(report.accuracy),
            "macro_precision": float(report.macro_precision),
            "macro_recall": float(report.macro_recall),
            "macro_f1": float(report.macro_f1),
            "weighted_precision": float(report.weighted_precision),
            "weighted_recall": float(report.weighted_recall),
            "weighted_f1": float(report.weighted_f1),
        },
    }

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "cfg": str(cfg_path.resolve()),
        "split": args.split,
        "feature": feature,
        "metrics": metrics,
        "artifacts": artifacts,
    }
    summary_path = args.out_dir / f"{run_name}_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Evaluation split={args.split} accuracy={accuracy:.4f} loss={loss:.5f}")
    print(f"Saved evaluation summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
