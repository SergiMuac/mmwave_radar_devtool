"""Train a neural network classifier from radar `.bin` captures."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..range_profile_dataset import parse_tensor_cfg
from .classification_eval import (
    pca_project_2d,
    plot_confidence_histogram,
    plot_confusion_matrix,
    plot_pca_scatter,
    plot_training_curves,
    save_classification_report_csv,
    save_confusion_matrix_csv,
    summarize_multiclass_predictions,
)
from .data import (
    BACKGROUND_SUBTRACTION_COMPLEX_RANGE,
    BACKGROUND_SUBTRACTION_NONE,
    BACKGROUND_SUBTRACTION_RAW,
    DEFAULT_FLAT_LABEL_REGEX,
    FEATURE_MODE_COMPLEX_COHERENT,
    FEATURE_MODE_ZERO_DOPPLER_DB,
    NORMALIZATION_NONE,
    NORMALIZATION_PER_RECORDING,
    NORMALIZATION_TRAINSET,
    BackgroundReference,
    LabeledCapture,
    apply_normalization_mode,
    discover_labeled_captures,
    extract_frame_feature_tensor_by_rx_db,
    feature_tensor_to_samples,
    load_background_reference,
    load_baseline_zero_doppler_mean_by_rx_db,
    parse_hidden_dims,
    split_captures_stratified,
)
from .models import RadarMLP


@dataclass(slots=True, frozen=True)
class ClassificationMetrics:
    """Aggregate classification metrics."""

    loss: float
    accuracy: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stack_split(
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
    background_reference: BackgroundReference | None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = parse_tensor_cfg(cfg_path)
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for capture in captures:
        frame_tensor = extract_frame_feature_tensor_by_rx_db(
            capture.path,
            cfg,
            range_side=range_side,
            window_kind=window_kind,
            eps=eps,
            max_frames=max_frames_per_capture,
            baseline_open_mean_db=baseline_open_mean_db,
            baseline_blocked_mean_db=baseline_blocked_mean_db,
            range_min_m=range_min_m,
            range_max_m=range_max_m,
            feature_mode=feature_mode,
            target_range_m=target_range_m,
            range_gate_bins=range_gate_bins,
            background_subtraction=background_subtraction,
            background_reference=background_reference,
        )
        if frame_tensor.shape[0] == 0:
            continue

        x = feature_tensor_to_samples(frame_tensor, sample_mode=sample_mode)
        y = np.full((x.shape[0],), class_to_index[capture.label], dtype=np.int64)
        all_x.append(x)
        all_y.append(y)

    if not all_x:
        raise RuntimeError("No samples were produced for this split.")

    return (
        np.concatenate(all_x, axis=0).astype(np.float32, copy=False),
        np.concatenate(all_y, axis=0).astype(np.int64, copy=False),
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
) -> ClassificationMetrics:
    model.eval()
    loss_sum = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = torch.argmax(logits, dim=1)

        n = int(y.numel())
        total += n
        loss_sum += float(loss.item()) * n
        correct += int((pred == y).sum().item())

    return ClassificationMetrics(
        loss=loss_sum / max(1, total),
        accuracy=correct / max(1, total),
    )


@torch.no_grad()
def _collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect logits and labels from one dataloader pass."""
    model.eval()
    logits_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        logits_batches.append(logits.detach().cpu().numpy())
        label_batches.append(y.detach().cpu().numpy())
    if not logits_batches:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return (
        np.concatenate(logits_batches, axis=0).astype(np.float32, copy=False),
        np.concatenate(label_batches, axis=0).astype(np.int64, copy=False),
    )


@torch.no_grad()
def _extract_model_embeddings(
    model: RadarMLP,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Extract penultimate embeddings for feature vectors `[N,D]`."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected [N,D] features, got {x.shape}")
    if x.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    ds = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(ds, batch_size=max(1, int(batch_size)), shuffle=False)

    model.eval()
    out: list[np.ndarray] = []
    for (x_batch,) in loader:
        emb = model.forward_features(x_batch.to(device))
        out.append(emb.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)


def _validate_split_disjointness(
    split_train: tuple[LabeledCapture, ...],
    split_val: tuple[LabeledCapture, ...],
    split_test: tuple[LabeledCapture, ...],
) -> None:
    """Ensure capture-level split disjointness to prevent leakage."""
    train_paths = {capture.path.resolve() for capture in split_train}
    val_paths = {capture.path.resolve() for capture in split_val}
    test_paths = {capture.path.resolve() for capture in split_test}

    if train_paths & val_paths:
        raise RuntimeError("Capture split leakage detected between train and val sets.")
    if train_paths & test_paths:
        raise RuntimeError("Capture split leakage detected between train and test sets.")
    if val_paths & test_paths:
        raise RuntimeError("Capture split leakage detected between val and test sets.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse classification training arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a radar neural-network classifier from `.bin` captures. "
            "Accepted dataset layouts: dataset/<label>/*.bin or <label>_<index>.bin"
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("config/xwr18xx_profile_raw_capture.cfg"),
        help="Radar cfg used to decode capture layout.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/ml/radar_classifier.pt"),
        help="Output checkpoint path.",
    )
    parser.add_argument("--epochs", "--max-epochs", dest="epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--hidden-dims",
        default="512,256",
        help="Comma-separated hidden dimensions, e.g. '512,256'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--early-stop-patience",
        "--early-stopping-patience",
        dest="early_stop_patience",
        type=int,
        default=10,
    )
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument(
        "--save-best-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, save/evaluate the best validation checkpoint (default: true).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate training/evaluation interpretability plots and CSV diagnostics.",
    )
    parser.add_argument(
        "--range-side",
        choices=["positive", "negative", "full"],
        default="positive",
        help="Range FFT side used for features.",
    )
    parser.add_argument("--window-kind", choices=["hann", "rect"], default="hann")
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument(
        "--range-min-m",
        type=float,
        default=None,
        help="Optional minimum target distance in meters for bin selection.",
    )
    parser.add_argument(
        "--range-max-m",
        type=float,
        default=None,
        help="Optional maximum target distance in meters for bin selection.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=[FEATURE_MODE_ZERO_DOPPLER_DB, FEATURE_MODE_COMPLEX_COHERENT],
        default=FEATURE_MODE_ZERO_DOPPLER_DB,
        help="Feature extraction mode.",
    )
    parser.add_argument(
        "--target-range-m",
        type=float,
        default=0.40,
        help="Target range center (meters) for complex_coherent gating.",
    )
    parser.add_argument(
        "--range-gate-bins",
        type=int,
        default=2,
        help="Half-width of target gate in bins for complex_coherent mode.",
    )
    parser.add_argument(
        "--background-subtraction",
        choices=[
            BACKGROUND_SUBTRACTION_NONE,
            BACKGROUND_SUBTRACTION_COMPLEX_RANGE,
            BACKGROUND_SUBTRACTION_RAW,
        ],
        default=None,
        help="Background subtraction mode for complex_coherent features.",
    )
    parser.add_argument(
        "--background-capture",
        type=Path,
        default=None,
        help="Optional empty-scene recording used for background subtraction.",
    )
    parser.add_argument(
        "--baseline-open-capture",
        type=Path,
        default=None,
        help=(
            "Optional no-attenuation baseline capture. "
            "If provided, train on frame-wise delta features."
        ),
    )
    parser.add_argument(
        "--baseline-blocked-capture",
        type=Path,
        default=None,
        help=(
            "Optional max-attenuation baseline capture. "
            "If provided, train on frame-wise delta features."
        ),
    )
    parser.add_argument(
        "--max-frames-per-capture",
        type=int,
        default=None,
        help="Optional cap on frames consumed from each capture.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["frames", "capture-mean", "capture-mean-std"],
        default="frames",
        help=(
            "How to turn frames inside each capture into training samples. "
            "'frames' keeps every frame; 'capture-mean' makes one averaged sample "
            "per capture; 'capture-mean-std' concatenates per-capture mean and std."
        ),
    )
    parser.add_argument(
        "--normalization-mode",
        choices=[NORMALIZATION_NONE, NORMALIZATION_PER_RECORDING, NORMALIZATION_TRAINSET],
        default=NORMALIZATION_TRAINSET,
        help="Feature normalization strategy applied before training.",
    )
    parser.add_argument(
        "--flat-label-regex",
        default=DEFAULT_FLAT_LABEL_REGEX,
        help="Regex used for flat `<label>_<index>.bin` datasets.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device selection.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate training arguments for predictable behavior."""
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError("--dropout must be in [0, 1)")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1)")
    if not (0.0 <= args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be in [0, 1)")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be < 1.0")
    if args.early_stop_patience <= 0:
        raise ValueError("--early-stop-patience must be > 0")
    if args.min_epochs <= 0:
        raise ValueError("--min-epochs must be > 0")
    if args.max_frames_per_capture is not None and args.max_frames_per_capture <= 0:
        raise ValueError("--max-frames-per-capture must be > 0 when set")
    if args.target_range_m < 0:
        raise ValueError("--target-range-m must be >= 0")
    if args.range_gate_bins < 0:
        raise ValueError("--range-gate-bins must be >= 0")
    if args.range_min_m is not None and args.range_min_m < 0:
        raise ValueError("--range-min-m must be >= 0 when set")
    if args.range_max_m is not None and args.range_max_m < 0:
        raise ValueError("--range-max-m must be >= 0 when set")
    if (
        args.range_min_m is not None
        and args.range_max_m is not None
        and args.range_min_m > args.range_max_m
    ):
        raise ValueError("--range-min-m must be <= --range-max-m when both are set")


def train_classifier(args: argparse.Namespace) -> dict[str, object]:
    """Run classification training from a parsed argument namespace."""
    _validate_args(args)

    _set_seed(int(args.seed))
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    cfg = parse_tensor_cfg(args.cfg)

    if args.background_subtraction is None:
        args.background_subtraction = (
            BACKGROUND_SUBTRACTION_COMPLEX_RANGE
            if args.feature_mode == FEATURE_MODE_COMPLEX_COHERENT
            else BACKGROUND_SUBTRACTION_NONE
        )

    if (
        args.feature_mode == FEATURE_MODE_COMPLEX_COHERENT
        and (args.baseline_open_capture is not None or args.baseline_blocked_capture is not None)
    ):
        raise ValueError(
            "--baseline-open-capture/--baseline-blocked-capture apply only to "
            "--feature-mode zero_doppler_db."
        )
    if (
        args.feature_mode == FEATURE_MODE_ZERO_DOPPLER_DB
        and args.background_subtraction != BACKGROUND_SUBTRACTION_NONE
    ):
        raise ValueError(
            "--background-subtraction is supported only for --feature-mode complex_coherent. "
            "Use --background-subtraction none for zero_doppler_db."
        )

    baseline_open_mean_db = None
    baseline_blocked_mean_db = None
    background_reference = None
    if args.feature_mode == FEATURE_MODE_ZERO_DOPPLER_DB:
        if args.baseline_open_capture is not None:
            baseline_open_mean_db = load_baseline_zero_doppler_mean_by_rx_db(
                args.baseline_open_capture,
                cfg,
                range_side=args.range_side,
                window_kind=args.window_kind,
                eps=float(args.eps),
                range_min_m=args.range_min_m,
                range_max_m=args.range_max_m,
            )
        if args.baseline_blocked_capture is not None:
            baseline_blocked_mean_db = load_baseline_zero_doppler_mean_by_rx_db(
                args.baseline_blocked_capture,
                cfg,
                range_side=args.range_side,
                window_kind=args.window_kind,
                eps=float(args.eps),
                range_min_m=args.range_min_m,
                range_max_m=args.range_max_m,
            )
    else:
        background_reference = load_background_reference(
            args.background_capture,
            cfg,
            window_kind=args.window_kind,
            eps=float(args.eps),
            background_subtraction=args.background_subtraction,
        )

    captures = discover_labeled_captures(
        args.dataset_dir,
        flat_label_regex=args.flat_label_regex,
    )
    labels = sorted({capture.label for capture in captures})
    if len(labels) < 2:
        raise RuntimeError(f"Need at least 2 classes, found {len(labels)}")

    split = split_captures_stratified(
        captures,
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    _validate_split_disjointness(split.train, split.val, split.test)

    if len(split.train) == 0 or len(split.val) == 0 or len(split.test) == 0:
        raise RuntimeError(
            "Capture split produced an empty split. Add more captures or lower val/test ratios."
        )

    class_to_index = {label: idx for idx, label in enumerate(labels)}
    print("Classes:", ", ".join(labels))
    print(
        "Capture split counts "
        f"train={len(split.train)} val={len(split.val)} test={len(split.test)}"
    )
    print("Split method: capture-level stratified (no frame/chirp leakage across splits)")
    if args.feature_mode == FEATURE_MODE_ZERO_DOPPLER_DB and (
        baseline_open_mean_db is not None or baseline_blocked_mean_db is not None
    ):
        print(
            "Feature mode: baseline delta "
            f"(open={args.baseline_open_capture is not None}, "
            f"blocked={args.baseline_blocked_capture is not None})"
        )
    elif args.feature_mode == FEATURE_MODE_COMPLEX_COHERENT:
        print(
            "Feature mode: complex coherent "
            f"(background_subtraction={args.background_subtraction})"
        )
    else:
        print("Feature mode: raw zero-Doppler dB")
    print(f"Sample mode: {args.sample_mode}")
    print(f"Normalization mode: {args.normalization_mode}")

    train_x, train_y = _stack_split(
        split.train,
        class_to_index=class_to_index,
        cfg_path=args.cfg,
        range_side=args.range_side,
        window_kind=args.window_kind,
        eps=float(args.eps),
        max_frames_per_capture=args.max_frames_per_capture,
        baseline_open_mean_db=baseline_open_mean_db,
        baseline_blocked_mean_db=baseline_blocked_mean_db,
        range_min_m=args.range_min_m,
        range_max_m=args.range_max_m,
        sample_mode=args.sample_mode,
        feature_mode=args.feature_mode,
        target_range_m=float(args.target_range_m),
        range_gate_bins=int(args.range_gate_bins),
        background_subtraction=args.background_subtraction,
        background_reference=background_reference,
    )
    val_x, val_y = _stack_split(
        split.val,
        class_to_index=class_to_index,
        cfg_path=args.cfg,
        range_side=args.range_side,
        window_kind=args.window_kind,
        eps=float(args.eps),
        max_frames_per_capture=args.max_frames_per_capture,
        baseline_open_mean_db=baseline_open_mean_db,
        baseline_blocked_mean_db=baseline_blocked_mean_db,
        range_min_m=args.range_min_m,
        range_max_m=args.range_max_m,
        sample_mode=args.sample_mode,
        feature_mode=args.feature_mode,
        target_range_m=float(args.target_range_m),
        range_gate_bins=int(args.range_gate_bins),
        background_subtraction=args.background_subtraction,
        background_reference=background_reference,
    )
    test_x, test_y = _stack_split(
        split.test,
        class_to_index=class_to_index,
        cfg_path=args.cfg,
        range_side=args.range_side,
        window_kind=args.window_kind,
        eps=float(args.eps),
        max_frames_per_capture=args.max_frames_per_capture,
        baseline_open_mean_db=baseline_open_mean_db,
        baseline_blocked_mean_db=baseline_blocked_mean_db,
        range_min_m=args.range_min_m,
        range_max_m=args.range_max_m,
        sample_mode=args.sample_mode,
        feature_mode=args.feature_mode,
        target_range_m=float(args.target_range_m),
        range_gate_bins=int(args.range_gate_bins),
        background_subtraction=args.background_subtraction,
        background_reference=background_reference,
    )

    train_x, val_x, test_x, mean, std = apply_normalization_mode(
        train_x,
        val_x,
        test_x,
        normalization_mode=args.normalization_mode,
    )

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = RadarMLP(
        input_dim=int(train_x.shape[1]),
        output_dim=len(labels),
        hidden_dims=hidden_dims,
        dropout=float(args.dropout),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_val_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    no_improve = 0

    history: list[dict[str, float | int]] = []
    final_train_metrics = ClassificationMetrics(loss=float("nan"), accuracy=float("nan"))
    final_val_metrics = ClassificationMetrics(loss=float("nan"), accuracy=float("nan"))

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            n = int(y_batch.numel())
            train_total += n
            train_loss_sum += float(loss.item()) * n
            train_correct += int((torch.argmax(logits, dim=1) == y_batch).sum().item())

        train_metrics = ClassificationMetrics(
            loss=train_loss_sum / max(1, train_total),
            accuracy=train_correct / max(1, train_total),
        )
        val_metrics = _evaluate(model, val_loader, device=device, criterion=criterion)
        final_train_metrics = train_metrics
        final_val_metrics = val_metrics

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_metrics.loss),
                "train_acc": float(train_metrics.accuracy),
                "val_loss": float(val_metrics.loss),
                "val_acc": float(val_metrics.accuracy),
            }
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics.loss:.5f} train_acc={train_metrics.accuracy:.4f} "
            f"val_loss={val_metrics.loss:.5f} val_acc={val_metrics.accuracy:.4f}"
        )

        improved = val_metrics.accuracy > best_val_acc or (
            abs(val_metrics.accuracy - best_val_acc) < 1e-12 and val_metrics.loss < best_val_loss
        )
        if improved:
            best_val_acc = val_metrics.accuracy
            best_val_loss = val_metrics.loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if epoch >= int(args.min_epochs) and no_improve >= int(args.early_stop_patience):
            print(
                "Early stopping "
                f"(best_epoch={best_epoch}, best_val_acc={best_val_acc:.4f}, "
                f"best_val_loss={best_val_loss:.5f})"
            )
            break

    if args.save_best_val and best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _evaluate(model, test_loader, device=device, criterion=criterion)
    print(f"test_loss={test_metrics.loss:.5f} test_acc={test_metrics.accuracy:.4f}")

    test_logits, test_true = _collect_logits_and_labels(model, test_loader, device=device)
    cm, test_pred, test_conf, correct_mask, class_report = summarize_multiclass_predictions(
        test_logits,
        test_true,
        class_names=labels,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_name = args.out.stem
    artifact_paths = {
        "training_curves": args.out.parent / f"{run_name}_training_curves.png",
        "confusion_matrix": args.out.parent / f"{run_name}_confusion_matrix.png",
        "confusion_matrix_normalized": args.out.parent
        / f"{run_name}_confusion_matrix_normalized.png",
        "confusion_matrix_csv": args.out.parent / f"{run_name}_confusion_matrix.csv",
        "confusion_matrix_normalized_csv": args.out.parent
        / f"{run_name}_confusion_matrix_normalized.csv",
        "classification_report_csv": args.out.parent / f"{run_name}_classification_report.csv",
        "pca_features": args.out.parent / f"{run_name}_pca_features.png",
        "pca_model_embedding": args.out.parent / f"{run_name}_pca_model_embedding.png",
        "confidence_histogram": args.out.parent / f"{run_name}_confidence_histogram.png",
    }

    save_confusion_matrix_csv(
        cm,
        class_names=labels,
        output_path=artifact_paths["confusion_matrix_csv"],
        normalized=False,
    )
    save_confusion_matrix_csv(
        cm,
        class_names=labels,
        output_path=artifact_paths["confusion_matrix_normalized_csv"],
        normalized=True,
    )
    save_classification_report_csv(
        class_report,
        output_path=artifact_paths["classification_report_csv"],
    )

    if args.plots:
        history_epochs = np.array([row["epoch"] for row in history], dtype=np.int32)
        history_train_loss = np.array([row["train_loss"] for row in history], dtype=np.float32)
        history_val_loss = np.array([row["val_loss"] for row in history], dtype=np.float32)
        history_train_acc = np.array([row["train_acc"] for row in history], dtype=np.float32)
        history_val_acc = np.array([row["val_acc"] for row in history], dtype=np.float32)

        plot_training_curves(
            epochs=history_epochs,
            train_loss=history_train_loss,
            val_loss=history_val_loss,
            train_acc=history_train_acc,
            val_acc=history_val_acc,
            output_path=artifact_paths["training_curves"],
        )
        plot_confusion_matrix(
            cm,
            class_names=labels,
            output_path=artifact_paths["confusion_matrix"],
            normalized=False,
        )
        plot_confusion_matrix(
            cm,
            class_names=labels,
            output_path=artifact_paths["confusion_matrix_normalized"],
            normalized=True,
        )

        pca_features_xy, pca_features_explained = pca_project_2d(test_x)
        plot_pca_scatter(
            pca_features_xy,
            y_true=test_true,
            y_pred=test_pred,
            class_names=labels,
            explained_variance=pca_features_explained,
            title="PCA of Input Features (Test Split)",
            output_path=artifact_paths["pca_features"],
        )

        model_embeddings = _extract_model_embeddings(
            model,
            test_x,
            device=device,
            batch_size=int(args.batch_size),
        )
        pca_embed_xy, pca_embed_explained = pca_project_2d(model_embeddings)
        plot_pca_scatter(
            pca_embed_xy,
            y_true=test_true,
            y_pred=test_pred,
            class_names=labels,
            explained_variance=pca_embed_explained,
            title="PCA of Learned Model Embedding (Test Split)",
            output_path=artifact_paths["pca_model_embedding"],
        )

        plot_confidence_histogram(
            test_conf,
            correct_mask=correct_mask,
            output_path=artifact_paths["confidence_histogram"],
        )

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "final_train_acc": float(final_train_metrics.accuracy),
        "final_val_acc": float(final_val_metrics.accuracy),
        "final_train_loss": float(final_train_metrics.loss),
        "final_val_loss": float(final_val_metrics.loss),
        "test_acc": float(test_metrics.accuracy),
        "test_loss": float(test_metrics.loss),
        "macro_precision": float(class_report.macro_precision),
        "macro_recall": float(class_report.macro_recall),
        "macro_f1": float(class_report.macro_f1),
        "weighted_precision": float(class_report.weighted_precision),
        "weighted_recall": float(class_report.weighted_recall),
        "weighted_f1": float(class_report.weighted_f1),
        "frame_counts": {
            "train": int(train_x.shape[0]),
            "val": int(val_x.shape[0]),
            "test": int(test_x.shape[0]),
        },
        "sample_counts": {
            "train": int(train_x.shape[0]),
            "val": int(val_x.shape[0]),
            "test": int(test_x.shape[0]),
        },
        "capture_counts": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        },
        "history": history,
        "classification_report": {
            "rows": list(class_report.rows),
            "accuracy": float(class_report.accuracy),
            "macro_precision": float(class_report.macro_precision),
            "macro_recall": float(class_report.macro_recall),
            "macro_f1": float(class_report.macro_f1),
            "weighted_precision": float(class_report.weighted_precision),
            "weighted_recall": float(class_report.weighted_recall),
            "weighted_f1": float(class_report.weighted_f1),
        },
    }

    checkpoint = {
        "model_state": model.state_dict(),
        "class_names": labels,
        "class_to_index": class_to_index,
        "input_dim": int(train_x.shape[1]),
        "hidden_dims": list(hidden_dims),
        "dropout": float(args.dropout),
        "feature": {
            "type": (
                "complex_coherent_reim_by_rx"
                if args.feature_mode == FEATURE_MODE_COMPLEX_COHERENT
                else (
                    "dual_delta_zero_doppler_by_rx_db_concat_bins"
                    if baseline_open_mean_db is not None and baseline_blocked_mean_db is not None
                    else (
                        "delta_zero_doppler_by_rx_db"
                        if baseline_open_mean_db is not None or baseline_blocked_mean_db is not None
                        else "zero_doppler_by_rx_db"
                    )
                )
            ),
            "feature_mode": args.feature_mode,
            "range_side": args.range_side,
            "window_kind": args.window_kind,
            "eps": float(args.eps),
            "cfg_path": str(args.cfg.resolve()),
            "max_frames_per_capture": args.max_frames_per_capture,
            "sample_mode": args.sample_mode,
            "normalization_mode": args.normalization_mode,
            "range_min_m": args.range_min_m,
            "range_max_m": args.range_max_m,
            "target_range_m": float(args.target_range_m),
            "range_gate_bins": int(args.range_gate_bins),
            "background_subtraction": args.background_subtraction,
            "background_capture": (
                None
                if args.background_capture is None
                else str(Path(args.background_capture).resolve())
            ),
            "baseline_open_capture": (
                None
                if args.baseline_open_capture is None
                else str(Path(args.baseline_open_capture).resolve())
            ),
            "baseline_blocked_capture": (
                None
                if args.baseline_blocked_capture is None
                else str(Path(args.baseline_blocked_capture).resolve())
            ),
        },
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "training": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "early_stop_patience": int(args.early_stop_patience),
            "min_epochs": int(args.min_epochs),
            "save_best_val": bool(args.save_best_val),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
            "flat_label_regex": str(args.flat_label_regex),
        },
        "metrics": metrics,
        "artifacts": {name: str(path.resolve()) for name, path in artifact_paths.items()},
    }
    torch.save(checkpoint, args.out)

    summary_path = args.out.with_suffix(".json")
    summary_payload = {
        "checkpoint": str(args.out.resolve()),
        "run_name": run_name,
        "class_names": labels,
        "class_to_index": class_to_index,
        "input_dim": int(train_x.shape[1]),
        "hidden_dims": list(hidden_dims),
        "dropout": float(args.dropout),
        "feature": checkpoint["feature"],
        "training": checkpoint["training"],
        "metrics": checkpoint["metrics"],
        "artifacts": checkpoint["artifacts"],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Saved checkpoint: {args.out}")
    print(f"Saved summary: {summary_path}")
    return summary_payload


def main(argv: Sequence[str] | None = None) -> int:
    """Run classification training."""
    args = parse_args(argv)
    train_classifier(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
