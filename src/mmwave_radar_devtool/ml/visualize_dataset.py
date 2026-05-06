"""Visualize average range-Doppler and range profiles over multiple captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..range_profile_dataset import compute_range_axis_m, parse_tensor_cfg, process_capture
from .data import (
    DEFAULT_FLAT_LABEL_REGEX,
    compute_average_range_doppler_by_rx,
    compute_average_range_profile_by_rx,
    discover_bin_files,
    infer_capture_label,
)


def _plot_range_doppler(
    range_doppler_by_rx: np.ndarray,
    *,
    title_prefix: str,
    output_path: Path,
    eps: float,
) -> None:
    rx_count = int(range_doppler_by_rx.shape[0])
    fig, axes = plt.subplots(1, rx_count, figsize=(6 * rx_count, 5), squeeze=False)

    for rx in range(rx_count):
        ax = axes[0, rx]
        db = 10.0 * np.log10(range_doppler_by_rx[rx] + float(eps))
        image = ax.imshow(db, aspect="auto", origin="lower")
        ax.set_title(f"{title_prefix} RX{rx}")
        ax.set_xlabel("Doppler bin")
        ax.set_ylabel("Range bin")
        fig.colorbar(image, ax=ax, shrink=0.85)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_range_profiles(
    range_profiles_by_rx: np.ndarray,
    *,
    title: str,
    output_path: Path,
    range_axis_m: np.ndarray | None,
    eps: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(range_profiles_by_rx.shape[1], dtype=np.float32)
    x_label = "Range bin"

    if range_axis_m is not None and range_axis_m.shape[0] == range_profiles_by_rx.shape[1]:
        x = range_axis_m.astype(np.float32)
        x_label = "Range (m)"

    for rx in range(range_profiles_by_rx.shape[0]):
        db = 10.0 * np.log10(range_profiles_by_rx[rx] + float(eps))
        ax.plot(x, db, label=f"RX{rx}")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Power (dB)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _average_receiver_response(range_profiles_by_rx: np.ndarray) -> np.ndarray:
    """Collapse `[RX, B]` receiver profiles into one class-average `[B]` curve."""
    arr = np.asarray(range_profiles_by_rx, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [RX, B] profile, got {arr.shape}")
    return np.mean(arr, axis=0, dtype=np.float32)


def _plot_class_average_profile(
    class_profile: np.ndarray,
    *,
    title: str,
    output_path: Path,
    range_axis_m: np.ndarray | None,
    eps: float,
) -> None:
    """Plot one receiver-collapsed class-average range profile."""
    profile = np.asarray(class_profile, dtype=np.float32)
    if profile.ndim != 1:
        raise ValueError(f"Expected [B] class profile, got {profile.shape}")

    x = np.arange(profile.shape[0], dtype=np.float32)
    x_label = "Range bin"
    if range_axis_m is not None and range_axis_m.shape[0] == profile.shape[0]:
        x = range_axis_m.astype(np.float32)
        x_label = "Range (m)"

    fig, ax = plt.subplots(figsize=(10, 5))
    db = 10.0 * np.log10(profile + float(eps))
    ax.plot(x, db, linewidth=2.4)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Power (dB)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_range_profile_overlays_by_label(
    profiles_by_label: dict[str, list[np.ndarray]],
    *,
    title: str,
    output_path: Path,
    range_axis_m: np.ndarray | None,
    eps: float,
) -> None:
    if not profiles_by_label:
        raise ValueError("profiles_by_label must not be empty")

    first_non_empty = next((profiles for profiles in profiles_by_label.values() if profiles), None)
    if first_non_empty is None:
        raise ValueError("profiles_by_label must contain at least one capture profile")

    sample_profile = np.asarray(first_non_empty[0], dtype=np.float32)
    if sample_profile.ndim != 2:
        raise ValueError(f"Expected [RX, B] profile, got {sample_profile.shape}")

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(sample_profile.shape[1], dtype=np.float32)
    x_label = "Range bin"
    if range_axis_m is not None and range_axis_m.shape[0] == sample_profile.shape[1]:
        x = range_axis_m.astype(np.float32)
        x_label = "Range (m)"

    cmap = plt.get_cmap("tab10" if len(profiles_by_label) <= 10 else "tab20")

    for label_idx, label in enumerate(sorted(profiles_by_label)):
        profiles = profiles_by_label[label]
        if not profiles:
            continue

        color = cmap(label_idx % cmap.N)
        class_profiles: list[np.ndarray] = []

        for profile in profiles:
            arr = np.asarray(profile, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Expected [RX, B] profile, got {arr.shape}")
            if arr.shape != sample_profile.shape:
                raise ValueError(
                    "All overlay profiles must share the same [RX, B] shape. "
                    f"expected={sample_profile.shape} got={arr.shape}"
                )

            class_profiles.append(_average_receiver_response(arr))

        label_mean = np.mean(np.stack(class_profiles, axis=0), axis=0)
        db = 10.0 * np.log10(label_mean + float(eps))
        ax.plot(x, db, color=color, linewidth=2.4, label=label)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Power (dB)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _normalize_range_axis(range_axis_m: np.ndarray, range_side: str) -> np.ndarray:
    if range_side == "positive":
        return range_axis_m[: range_axis_m.shape[0] // 2]
    if range_side == "negative":
        return range_axis_m[range_axis_m.shape[0] // 2 :]
    return range_axis_m


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate average Doppler-range data by receiver over a dataset of `.bin` captures."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("config/xwr18xx_profile_raw_capture.cfg"),
        help="Radar cfg used to decode capture layout.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ml_visuals"))
    parser.add_argument("--window-kind", choices=["hann", "rect"], default="hann")
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument(
        "--range-side",
        choices=["positive", "negative", "full"],
        default="positive",
        help="Range FFT side shown in plots.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of captures to aggregate.",
    )
    parser.add_argument(
        "--group-by-label",
        action="store_true",
        help="Also save per-label averaged plots/arrays.",
    )
    parser.add_argument(
        "--plot-run-overlays",
        action="store_true",
        help="Also save per-label plots with all capture runs overlaid.",
    )
    parser.add_argument(
        "--flat-label-regex",
        default=DEFAULT_FLAT_LABEL_REGEX,
        help="Regex used for flat `<label>_<index>.bin` datasets.",
    )
    return parser.parse_args()


def main() -> int:
    """Aggregate and plot dataset-level statistics."""
    args = parse_args()
    if args.plot_run_overlays and not args.group_by_label:
        raise ValueError("--plot-run-overlays requires --group-by-label")

    captures = discover_bin_files(args.dataset_dir)
    if not captures:
        raise RuntimeError(f"No `.bin` captures found under {args.dataset_dir}")

    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("--max-files must be > 0 when set")
        captures = captures[: int(args.max_files)]

    cfg = parse_tensor_cfg(args.cfg)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    global_sum: np.ndarray | None = None
    global_count = 0

    label_sums: dict[str, np.ndarray] = {}
    label_counts: dict[str, int] = {}
    label_capture_profiles: dict[str, list[np.ndarray]] = {}

    for idx, capture_path in enumerate(captures, start=1):
        processed = process_capture(
            capture_path,
            cfg,
            window_kind=args.window_kind,
            eps=float(args.eps),
            allow_trailing_partial_iq=True,
            allow_trailing_partial_frame=True,
        )
        rd_by_rx = compute_average_range_doppler_by_rx(processed, range_side=args.range_side)
        rp_by_rx = compute_average_range_profile_by_rx(rd_by_rx)

        if global_sum is None:
            global_sum = np.zeros_like(rd_by_rx, dtype=np.float64)
        global_sum += rd_by_rx.astype(np.float64)
        global_count += 1

        if args.group_by_label:
            label = infer_capture_label(
                capture_path,
                args.dataset_dir,
                flat_label_regex=args.flat_label_regex,
            )
            if label is None:
                label = "unlabeled"

            if label not in label_sums:
                label_sums[label] = np.zeros_like(rd_by_rx, dtype=np.float64)
                label_counts[label] = 0
                label_capture_profiles[label] = []
            label_sums[label] += rd_by_rx.astype(np.float64)
            label_counts[label] += 1
            if args.plot_run_overlays:
                label_capture_profiles[label].append(rp_by_rx)

        print(f"Processed {idx}/{len(captures)}: {capture_path.name}")

    if global_sum is None or global_count == 0:
        raise RuntimeError("No captures were successfully processed.")

    avg_rd = (global_sum / float(global_count)).astype(np.float32)
    avg_rp = compute_average_range_profile_by_rx(avg_rd)

    range_axis_m = _normalize_range_axis(compute_range_axis_m(cfg), args.range_side)

    global_rd_png = output_dir / "dataset_avg_range_doppler_by_rx.png"
    global_rp_png = output_dir / "dataset_avg_range_profile_by_rx.png"
    global_rd_npy = output_dir / "dataset_avg_range_doppler_by_rx.npy"
    global_rp_npy = output_dir / "dataset_avg_range_profile_by_rx.npy"

    np.save(global_rd_npy, avg_rd)
    np.save(global_rp_npy, avg_rp)
    _plot_range_doppler(
        avg_rd,
        title_prefix="Dataset avg",
        output_path=global_rd_png,
        eps=float(args.eps),
    )
    _plot_range_profiles(
        avg_rp,
        title="Dataset average range profile by receiver",
        output_path=global_rp_png,
        range_axis_m=range_axis_m,
        eps=float(args.eps),
    )

    per_label_outputs: dict[str, dict[str, str]] = {}
    if args.group_by_label:
        for label in sorted(label_sums):
            avg_label_rd = (label_sums[label] / float(label_counts[label])).astype(np.float32)
            avg_label_rp_by_rx = compute_average_range_profile_by_rx(avg_label_rd)
            avg_label_rp = _average_receiver_response(avg_label_rp_by_rx)

            safe_label = label.replace("/", "_")
            rd_png = output_dir / f"label_{safe_label}_avg_range_doppler_by_rx.png"
            rp_png = output_dir / f"label_{safe_label}_avg_range_profile_by_rx.png"
            rd_npy = output_dir / f"label_{safe_label}_avg_range_doppler_by_rx.npy"
            rp_npy = output_dir / f"label_{safe_label}_avg_range_profile_by_rx.npy"

            np.save(rd_npy, avg_label_rd)
            np.save(rp_npy, avg_label_rp)
            _plot_range_doppler(
                avg_label_rd,
                title_prefix=f"{label} avg",
                output_path=rd_png,
                eps=float(args.eps),
            )
            _plot_range_profiles(
                avg_label_rp_by_rx,
                title=f"{label} average range profile by receiver",
                output_path=rp_png,
                range_axis_m=range_axis_m,
                eps=float(args.eps),
            )

            overlay_png = None
            per_label_outputs[label] = {
                "captures": str(label_counts[label]),
                "range_doppler_png": str(rd_png.resolve()),
                "range_profile_png": str(rp_png.resolve()),
                "range_doppler_npy": str(rd_npy.resolve()),
                "range_profile_npy": str(rp_npy.resolve()),
            }

    class_overlay_png = None
    if args.plot_run_overlays and label_capture_profiles:
        class_overlay_png = output_dir / "dataset_class_average_overlay_range_profile.png"
        _plot_range_profile_overlays_by_label(
            label_capture_profiles,
            title="Class average range profile overlay",
            output_path=class_overlay_png,
            range_axis_m=range_axis_m,
            eps=float(args.eps),
        )

    summary = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "cfg": str(args.cfg.resolve()),
        "range_side": args.range_side,
        "window_kind": args.window_kind,
        "captures_processed": int(global_count),
        "rx_count": int(avg_rd.shape[0]),
        "range_bins": int(avg_rd.shape[1]),
        "doppler_bins": int(avg_rd.shape[2]),
        "outputs": {
            "range_doppler_png": str(global_rd_png.resolve()),
            "range_profile_png": str(global_rp_png.resolve()),
            "range_doppler_npy": str(global_rd_npy.resolve()),
            "range_profile_npy": str(global_rp_npy.resolve()),
            "class_average_overlay_png": (
                None if class_overlay_png is None else str(class_overlay_png.resolve())
            ),
        },
        "per_label_outputs": per_label_outputs,
    }
    summary_path = output_dir / "dataset_visual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {global_rd_png}")
    print(f"Saved: {global_rp_png}")
    print(f"Saved: {global_rd_npy}")
    print(f"Saved: {global_rp_npy}")
    print(f"Saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
