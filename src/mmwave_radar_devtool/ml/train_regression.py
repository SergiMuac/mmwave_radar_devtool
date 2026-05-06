"""Train a neural network regressor from radar `.bin` captures."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..range_profile_dataset import parse_tensor_cfg
from .data import (
    BACKGROUND_SUBTRACTION_COMPLEX_RANGE,
    BACKGROUND_SUBTRACTION_NONE,
    BACKGROUND_SUBTRACTION_RAW,
    DEFAULT_FLAT_LABEL_REGEX,
    DEFAULT_NUMERIC_TARGET_REGEX,
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
    load_label_target_map,
    parse_hidden_dims,
    resolve_regression_target,
    split_captures_stratified,
)
from .models import RadarMLP


@dataclass(slots=True, frozen=True)
class RegressionMetrics:
    """Aggregate regression metrics."""

    mse: float
    rmse: float
    mae: float
    r2: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
) -> RegressionMetrics:
    model.eval()
    total = 0
    mse_sum = 0.0
    mae_sum = 0.0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).squeeze(1)
        loss = criterion(pred, y)

        n = int(y.numel())
        total += n
        mse_sum += float(loss.item()) * n
        mae_sum += float(torch.mean(torch.abs(pred - y)).item()) * n

        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())

    pred_all = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
    target_all = np.concatenate(targets, axis=0) if targets else np.zeros((0,), dtype=np.float32)

    mse = mse_sum / max(1, total)
    rmse = math.sqrt(mse)
    mae = mae_sum / max(1, total)

    if target_all.size == 0:
        r2 = float("nan")
    else:
        ss_res = float(np.sum((pred_all - target_all) ** 2))
        target_mean = float(np.mean(target_all))
        ss_tot = float(np.sum((target_all - target_mean) ** 2))
        r2 = float("nan") if ss_tot <= 0.0 else 1.0 - (ss_res / ss_tot)

    return RegressionMetrics(mse=mse, rmse=rmse, mae=mae, r2=r2)


def _stack_split(
    captures: tuple[LabeledCapture, ...],
    *,
    target_by_capture: dict[Path, float],
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
        y = np.full((x.shape[0],), float(target_by_capture[capture.path]), dtype=np.float32)
        all_x.append(x)
        all_y.append(y)

    if not all_x:
        raise RuntimeError("No samples were produced for this split.")

    return (
        np.concatenate(all_x, axis=0).astype(np.float32, copy=False),
        np.concatenate(all_y, axis=0).astype(np.float32, copy=False),
    )


def parse_args() -> argparse.Namespace:
    """Parse regression training arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a radar neural-network regressor from `.bin` captures. "
            "Targets can come from label map JSON or parsed numeric labels."
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
        default=Path("outputs/ml/radar_regressor.pt"),
        help="Output checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=40)
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
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=8)
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
        "--numeric-target-regex",
        default=DEFAULT_NUMERIC_TARGET_REGEX,
        help="Regex used to parse numeric target from label/stem when map is absent.",
    )
    parser.add_argument(
        "--label-target-map",
        type=Path,
        default=None,
        help="Optional JSON map {\"label\": numeric_target}.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device selection.",
    )
    return parser.parse_args()


def main() -> int:
    """Run regression training."""
    args = parse_args()

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
    label_target_map = load_label_target_map(args.label_target_map)

    target_by_capture: dict[Path, float] = {}
    for capture in captures:
        target_by_capture[capture.path] = resolve_regression_target(
            capture,
            label_target_map=label_target_map,
            numeric_target_regex=args.numeric_target_regex,
        )

    split = split_captures_stratified(
        captures,
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )

    if len(split.train) == 0 or len(split.val) == 0 or len(split.test) == 0:
        raise RuntimeError(
            "Capture split produced an empty split. Add more captures or lower val/test ratios."
        )

    print(
        "Capture split counts "
        f"train={len(split.train)} val={len(split.val)} test={len(split.test)}"
    )
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
        target_by_capture=target_by_capture,
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
        target_by_capture=target_by_capture,
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
        target_by_capture=target_by_capture,
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
        output_dim=1,
        hidden_dims=hidden_dims,
        dropout=float(args.dropout),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    criterion = nn.MSELoss()

    best_val_rmse = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_mse_sum = 0.0
        train_mae_sum = 0.0
        train_total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_batch).squeeze(1)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            n = int(y_batch.numel())
            train_total += n
            train_mse_sum += float(loss.item()) * n
            train_mae_sum += float(torch.mean(torch.abs(pred - y_batch)).item()) * n

        train_mse = train_mse_sum / max(1, train_total)
        train_rmse = math.sqrt(train_mse)
        train_mae = train_mae_sum / max(1, train_total)

        val_metrics = _evaluate(model, val_loader, device=device, criterion=criterion)
        print(
            f"epoch={epoch:03d} "
            f"train_rmse={train_rmse:.5f} train_mae={train_mae:.5f} "
            f"val_rmse={val_metrics.rmse:.5f} "
            f"val_mae={val_metrics.mae:.5f} val_r2={val_metrics.r2:.5f}"
        )

        if val_metrics.rmse < best_val_rmse:
            best_val_rmse = val_metrics.rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if epoch >= int(args.min_epochs) and no_improve >= int(args.early_stop_patience):
            print(
                "Early stopping "
                f"(best_epoch={best_epoch}, best_val_rmse={best_val_rmse:.5f})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _evaluate(model, test_loader, device=device, criterion=criterion)
    print(
        f"test_rmse={test_metrics.rmse:.5f} test_mae={test_metrics.mae:.5f} "
        f"test_r2={test_metrics.r2:.5f}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
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
                None if args.background_capture is None else str(args.background_capture.resolve())
            ),
            "baseline_open_capture": (
                None
                if args.baseline_open_capture is None
                else str(args.baseline_open_capture.resolve())
            ),
            "baseline_blocked_capture": (
                None
                if args.baseline_blocked_capture is None
                else str(args.baseline_blocked_capture.resolve())
            ),
        },
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "targets": {
            str(path.resolve()): float(value)
            for path, value in sorted(target_by_capture.items(), key=lambda item: str(item[0]))
        },
        "metrics": {
            "best_epoch": int(best_epoch),
            "best_val_rmse": float(best_val_rmse),
            "test_rmse": float(test_metrics.rmse),
            "test_mae": float(test_metrics.mae),
            "test_r2": float(test_metrics.r2),
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
        },
    }
    torch.save(checkpoint, args.out)

    summary_path = args.out.with_suffix(".json")
    summary_payload = {
        "checkpoint": str(args.out.resolve()),
        "input_dim": int(train_x.shape[1]),
        "hidden_dims": list(hidden_dims),
        "dropout": float(args.dropout),
        "feature": checkpoint["feature"],
        "metrics": checkpoint["metrics"],
        "target_source": {
            "label_target_map": (
                None if args.label_target_map is None else str(args.label_target_map.resolve())
            ),
            "numeric_target_regex": args.numeric_target_regex,
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Saved checkpoint: {args.out}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
