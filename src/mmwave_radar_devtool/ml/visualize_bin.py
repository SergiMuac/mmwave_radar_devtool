"""Visualize average range-Doppler and range profiles for one `.bin` capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..range_profile_dataset import compute_range_axis_m, parse_tensor_cfg, process_capture
from .data import compute_average_range_doppler_by_rx, compute_average_range_profile_by_rx


def _plot_range_doppler(
    range_doppler_by_rx: np.ndarray,
    *,
    output_path: Path,
    eps: float,
) -> None:
    rx_count = int(range_doppler_by_rx.shape[0])
    fig, axes = plt.subplots(1, rx_count, figsize=(6 * rx_count, 5), squeeze=False)

    for rx in range(rx_count):
        ax = axes[0, rx]
        db = 10.0 * np.log10(range_doppler_by_rx[rx] + float(eps))
        image = ax.imshow(db, aspect="auto", origin="lower")
        ax.set_title(f"RX{rx} avg range-doppler")
        ax.set_xlabel("Doppler bin")
        ax.set_ylabel("Range bin")
        fig.colorbar(image, ax=ax, shrink=0.85)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_range_profiles(
    range_profiles_by_rx: np.ndarray,
    *,
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

    ax.set_title("Average range profile by receiver")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Power (dB)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize average Doppler-range data by receiver for one `.bin` capture."
    )
    parser.add_argument("capture", type=Path)
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
    return parser.parse_args()


def main() -> int:
    """Generate plots and arrays for one capture."""
    args = parse_args()
    cfg = parse_tensor_cfg(args.cfg)
    processed = process_capture(
        args.capture,
        cfg,
        window_kind=args.window_kind,
        eps=float(args.eps),
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    range_doppler_by_rx = compute_average_range_doppler_by_rx(
        processed,
        range_side=args.range_side,
    )
    range_profiles_by_rx = compute_average_range_profile_by_rx(range_doppler_by_rx)

    range_axis_m = compute_range_axis_m(cfg)
    if args.range_side == "positive":
        range_axis_m = range_axis_m[: range_axis_m.shape[0] // 2]
    elif args.range_side == "negative":
        range_axis_m = range_axis_m[range_axis_m.shape[0] // 2 :]

    stem = args.capture.stem
    rd_png = output_dir / f"{stem}_avg_range_doppler_by_rx.png"
    rp_png = output_dir / f"{stem}_avg_range_profile_by_rx.png"
    rd_npy = output_dir / f"{stem}_avg_range_doppler_by_rx.npy"
    rp_npy = output_dir / f"{stem}_avg_range_profile_by_rx.npy"
    summary_json = output_dir / f"{stem}_summary.json"

    np.save(rd_npy, range_doppler_by_rx.astype(np.float32))
    np.save(rp_npy, range_profiles_by_rx.astype(np.float32))

    _plot_range_doppler(range_doppler_by_rx, output_path=rd_png, eps=float(args.eps))
    _plot_range_profiles(
        range_profiles_by_rx,
        output_path=rp_png,
        range_axis_m=range_axis_m,
        eps=float(args.eps),
    )

    summary = {
        "capture": str(args.capture.resolve()),
        "cfg": str(args.cfg.resolve()),
        "range_side": args.range_side,
        "window_kind": args.window_kind,
        "frames": int(processed.cube.shape[0]),
        "rx_count": int(range_doppler_by_rx.shape[0]),
        "range_bins": int(range_doppler_by_rx.shape[1]),
        "doppler_bins": int(range_doppler_by_rx.shape[2]),
        "outputs": {
            "range_doppler_png": str(rd_png.resolve()),
            "range_profile_png": str(rp_png.resolve()),
            "range_doppler_npy": str(rd_npy.resolve()),
            "range_profile_npy": str(rp_npy.resolve()),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {rd_png}")
    print(f"Saved: {rp_png}")
    print(f"Saved: {rd_npy}")
    print(f"Saved: {rp_npy}")
    print(f"Saved: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
