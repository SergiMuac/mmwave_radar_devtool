from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmwave_radar_devtool.range_profile_dataset import (
    RadarTensorConfig,
    compute_range_axis_m,
    create_training_inputs,
    detect_target_bin_candidates,
    parse_tensor_cfg,
    plot_range_doppler_heatmap,
    plot_range_profile_slice,
    plot_zero_doppler_average,
    process_capture,
    select_useful_range_side,
    validate_capture_layout,
)


def _default_tensor_config() -> RadarTensorConfig:
    """Return the default xWR18xx tensor configuration from this project context."""
    return RadarTensorConfig(
        chirps_per_frame=16,
        num_rx=4,
        num_tx=1,
        num_adc_samples=256,
        adc_sample_rate_ksps=6000.0,
        freq_slope_mhz_per_us=80.0,
        start_freq_ghz=77.0,
        idle_time_us=10.0,
        ramp_end_time_us=50.0,
    )


def _resolve_config(args: argparse.Namespace) -> RadarTensorConfig:
    """Build the tensor configuration from cfg and optional overrides."""
    cfg = parse_tensor_cfg(args.cfg) if args.cfg is not None else _default_tensor_config()

    updates = {}
    if args.chirps_per_frame is not None:
        updates["chirps_per_frame"] = args.chirps_per_frame
    if args.num_rx is not None:
        updates["num_rx"] = args.num_rx
    if args.num_tx is not None:
        updates["num_tx"] = args.num_tx
    if args.num_adc_samples is not None:
        updates["num_adc_samples"] = args.num_adc_samples
    if args.adc_sample_rate_ksps is not None:
        updates["adc_sample_rate_ksps"] = args.adc_sample_rate_ksps
    if args.freq_slope_mhz_per_us is not None:
        updates["freq_slope_mhz_per_us"] = args.freq_slope_mhz_per_us
    if args.start_freq_ghz is not None:
        updates["start_freq_ghz"] = args.start_freq_ghz
    if args.idle_time_us is not None:
        updates["idle_time_us"] = args.idle_time_us
    if args.ramp_end_time_us is not None:
        updates["ramp_end_time_us"] = args.ramp_end_time_us

    if updates:
        cfg = replace(cfg, **updates)
    return cfg


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Process TI xWR18xx/xWR68xx DCA1000 raw ADC captures into range-profile "
            "tensors for model training."
        )
    )
    parser.add_argument(
        "object_capture",
        nargs="?",
        type=Path,
        default=Path("object_capture.bin"),
        help="Path to object capture .bin (default: object_capture.bin).",
    )
    parser.add_argument(
        "--empty-capture",
        type=Path,
        default=None,
        help="Optional empty/background capture .bin for subtraction.",
    )
    parser.add_argument(
        "--baseline-open-capture",
        type=Path,
        default=None,
        help=(
            "Optional no-attenuation baseline .bin. "
            "When provided, saves delta tensors/plots versus this baseline."
        ),
    )
    parser.add_argument(
        "--baseline-blocked-capture",
        type=Path,
        default=None,
        help=(
            "Optional max-attenuation baseline .bin. "
            "When provided, saves delta tensors/plots versus this baseline."
        ),
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("config/xwr18xx_profile_raw_capture.cfg"),
        help="TI cfg used to derive tensor dimensions and profile parameters.",
    )
    parser.add_argument("--chirps-per-frame", type=int, default=None)
    parser.add_argument("--num-rx", type=int, default=None)
    parser.add_argument("--num-tx", type=int, default=None)
    parser.add_argument("--num-adc-samples", type=int, default=None)
    parser.add_argument("--adc-sample-rate-ksps", type=float, default=None)
    parser.add_argument("--freq-slope-mhz-per-us", type=float, default=None)
    parser.add_argument("--start-freq-ghz", type=float, default=None)
    parser.add_argument("--idle-time-us", type=float, default=None)
    parser.add_argument("--ramp-end-time-us", type=float, default=None)
    parser.add_argument("--window-kind", choices=["hann", "rect"], default="hann")
    parser.add_argument("--window-frames", type=int, default=None)
    parser.add_argument("--window-step", type=int, default=1)
    parser.add_argument("--target-m", type=float, default=0.4)
    parser.add_argument("--frame-for-plots", type=int, default=0)
    parser.add_argument(
        "--useful-side",
        choices=["full", "positive", "negative"],
        default="full",
        help="Select positive/negative FFT half after inspection for NN output.",
    )
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def _select_range_axis_side(range_axis_m: np.ndarray, *, side: str) -> np.ndarray:
    """Select range-axis half matching useful-side selection."""
    if side == "positive":
        return range_axis_m[: range_axis_m.shape[0] // 2]
    if side == "negative":
        return range_axis_m[range_axis_m.shape[0] // 2 :]
    return range_axis_m


def _delta_against_baseline(object_logmag: np.ndarray, baseline_logmag: np.ndarray) -> np.ndarray:
    """Subtract baseline mean frame from object tensor."""
    baseline_mean = np.mean(np.asarray(baseline_logmag, dtype=np.float32), axis=0, keepdims=True)
    return (np.asarray(object_logmag, dtype=np.float32) - baseline_mean).astype(np.float32)


def _plot_baseline_delta_profiles(
    *,
    delta_vs_open: np.ndarray | None,
    delta_vs_blocked: np.ndarray | None,
    output_path: Path,
    range_axis_m: np.ndarray | None,
) -> None:
    """Plot side-by-side average delta profiles for baseline comparisons."""
    traces: list[tuple[str, np.ndarray]] = []
    if delta_vs_open is not None:
        open_profile = np.mean(delta_vs_open, axis=(0, 1, 2), dtype=np.float32)
        traces.append(
            ("Delta vs no-attenuation baseline", open_profile)
        )
    if delta_vs_blocked is not None:
        traces.append(
            (
                "Delta vs max-attenuation baseline",
                np.mean(delta_vs_blocked, axis=(0, 1, 2), dtype=np.float32),
            )
        )
    if not traces:
        return

    fig, axes = plt.subplots(
        1,
        len(traces),
        figsize=(7 * len(traces), 5),
        squeeze=False,
        sharey=True,
    )
    x_label = "Range FFT Bin"
    for idx, (title, y_values) in enumerate(traces):
        x = np.arange(y_values.size, dtype=np.float32)
        if range_axis_m is not None and range_axis_m.shape[0] == y_values.size:
            x = range_axis_m.astype(np.float32)
            x_label = "Range (m)"
        ax = axes[0, idx]
        ax.plot(x, y_values, linewidth=1.4)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        if idx == 0:
            ax.set_ylabel("Delta log-magnitude (dB)")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Example entry point producing .npy tensors and debug plots."""
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _resolve_config(args)

    print("Tensor config")
    print(f"  chirps_per_frame={cfg.chirps_per_frame}")
    print(f"  num_rx={cfg.num_rx}")
    print(f"  num_tx={cfg.num_tx}")
    print(f"  num_adc_samples={cfg.num_adc_samples}")
    print(f"  adc_sample_rate_ksps={cfg.adc_sample_rate_ksps}")
    print(f"  freq_slope_mhz_per_us={cfg.freq_slope_mhz_per_us}")
    print(f"  expected_bytes_per_frame={cfg.bytes_per_frame}")

    validate_capture_layout(args.object_capture, cfg)
    object_data = process_capture(
        args.object_capture,
        cfg,
        window_kind=args.window_kind,
        eps=args.eps,
    )

    empty_data = None
    if args.empty_capture is not None:
        validate_capture_layout(args.empty_capture, cfg)
        empty_data = process_capture(
            args.empty_capture,
            cfg,
            window_kind=args.window_kind,
            eps=args.eps,
        )

    baseline_open_data = None
    if args.baseline_open_capture is not None:
        validate_capture_layout(args.baseline_open_capture, cfg)
        baseline_open_data = process_capture(
            args.baseline_open_capture,
            cfg,
            window_kind=args.window_kind,
            eps=args.eps,
        )

    baseline_blocked_data = None
    if args.baseline_blocked_capture is not None:
        validate_capture_layout(args.baseline_blocked_capture, cfg)
        baseline_blocked_data = process_capture(
            args.baseline_blocked_capture,
            cfg,
            window_kind=args.window_kind,
            eps=args.eps,
        )

    zero_doppler_db = object_data.zero_doppler_db
    rd_heatmap = object_data.range_doppler_power_shifted

    # Optional side selection for full-complex FFT bins.
    object_logmag_for_nn = object_data.log_magnitude_db
    empty_logmag_for_nn = empty_data.log_magnitude_db if empty_data is not None else None
    baseline_open_logmag = (
        baseline_open_data.log_magnitude_db if baseline_open_data is not None else None
    )
    baseline_blocked_logmag = (
        baseline_blocked_data.log_magnitude_db if baseline_blocked_data is not None else None
    )
    if args.useful_side != "full":
        object_logmag_for_nn = select_useful_range_side(
            object_logmag_for_nn, side=args.useful_side
        )
        if empty_logmag_for_nn is not None:
            empty_logmag_for_nn = select_useful_range_side(
                empty_logmag_for_nn, side=args.useful_side
            )
        if baseline_open_logmag is not None:
            baseline_open_logmag = select_useful_range_side(
                baseline_open_logmag, side=args.useful_side
            )
        if baseline_blocked_logmag is not None:
            baseline_blocked_logmag = select_useful_range_side(
                baseline_blocked_logmag, side=args.useful_side
            )

    nn_logmag_windows = create_training_inputs(
        object_logmag_for_nn,
        empty_logmag=empty_logmag_for_nn,
        window_frames=args.window_frames,
        window_step=args.window_step,
    )

    delta_vs_open = None
    if baseline_open_logmag is not None:
        delta_vs_open = _delta_against_baseline(object_logmag_for_nn, baseline_open_logmag)

    delta_vs_blocked = None
    if baseline_blocked_logmag is not None:
        delta_vs_blocked = _delta_against_baseline(object_logmag_for_nn, baseline_blocked_logmag)

    np.save(output_dir / "range_profile_zero_doppler.npy", zero_doppler_db)
    np.save(output_dir / "range_doppler_heatmap.npy", rd_heatmap)
    np.save(output_dir / "nn_logmag_windows.npy", nn_logmag_windows)
    if delta_vs_open is not None:
        np.save(output_dir / "delta_vs_open_baseline_logmag.npy", delta_vs_open)
    if delta_vs_blocked is not None:
        np.save(output_dir / "delta_vs_blocked_baseline_logmag.npy", delta_vs_blocked)

    # Debug plots
    plot_frame = max(0, min(args.frame_for_plots, object_data.range_cube.shape[0] - 1))
    plot_range_profile_slice(
        object_data.range_cube,
        frame_idx=plot_frame,
        chirp_idx=0,
        rx_idx=0,
        output_path=output_dir / "debug_range_profile_slice.png",
        eps=args.eps,
    )

    range_axis_m = compute_range_axis_m(cfg)
    selected_range_axis = _select_range_axis_side(range_axis_m, side=args.useful_side)
    plot_zero_doppler_average(
        zero_doppler_db,
        output_path=output_dir / "debug_zero_doppler_profile_avg.png",
        range_axis_m=range_axis_m,
    )

    plot_range_doppler_heatmap(
        object_data.range_doppler_power_unshifted,
        frame_idx=plot_frame,
        output_path=output_dir / "debug_range_doppler_unshifted.png",
        eps=args.eps,
    )
    plot_range_doppler_heatmap(
        object_data.range_doppler_power_shifted,
        frame_idx=plot_frame,
        output_path=output_dir / "debug_range_doppler_shifted.png",
        eps=args.eps,
    )

    # Target-bin inspection utility around 40 cm by default.
    mean_zero_doppler = np.mean(zero_doppler_db, axis=0)
    target_report = detect_target_bin_candidates(
        mean_zero_doppler,
        cfg,
        target_m=args.target_m,
        search_radius=2,
    )
    (output_dir / "target_bin_report.json").write_text(
        json.dumps(target_report, indent=2),
        encoding="utf-8",
    )

    delta_plot_path = output_dir / "debug_delta_baseline_profiles.png"
    _plot_baseline_delta_profiles(
        delta_vs_open=delta_vs_open,
        delta_vs_blocked=delta_vs_blocked,
        output_path=delta_plot_path,
        range_axis_m=selected_range_axis,
    )

    delta_summary = {
        "useful_side": args.useful_side,
        "has_open_baseline": delta_vs_open is not None,
        "has_blocked_baseline": delta_vs_blocked is not None,
    }
    if delta_vs_open is not None:
        delta_summary["delta_vs_open_mean_db"] = float(np.mean(delta_vs_open))
        delta_summary["delta_vs_open_std_db"] = float(np.std(delta_vs_open))
    if delta_vs_blocked is not None:
        delta_summary["delta_vs_blocked_mean_db"] = float(np.mean(delta_vs_blocked))
        delta_summary["delta_vs_blocked_std_db"] = float(np.std(delta_vs_blocked))
    if delta_vs_open is not None or delta_vs_blocked is not None:
        (output_dir / "delta_baseline_summary.json").write_text(
            json.dumps(delta_summary, indent=2),
            encoding="utf-8",
        )

    print("Saved outputs")
    print(f"  {output_dir / 'range_profile_zero_doppler.npy'}")
    print(f"  {output_dir / 'range_doppler_heatmap.npy'}")
    print(f"  {output_dir / 'nn_logmag_windows.npy'}")
    print(f"  {output_dir / 'debug_range_profile_slice.png'}")
    print(f"  {output_dir / 'debug_zero_doppler_profile_avg.png'}")
    print(f"  {output_dir / 'debug_range_doppler_unshifted.png'}")
    print(f"  {output_dir / 'debug_range_doppler_shifted.png'}")
    print(f"  {output_dir / 'target_bin_report.json'}")
    if delta_vs_open is not None:
        print(f"  {output_dir / 'delta_vs_open_baseline_logmag.npy'}")
    if delta_vs_blocked is not None:
        print(f"  {output_dir / 'delta_vs_blocked_baseline_logmag.npy'}")
    if delta_vs_open is not None or delta_vs_blocked is not None:
        print(f"  {output_dir / 'debug_delta_baseline_profiles.png'}")
        print(f"  {output_dir / 'delta_baseline_summary.json'}")
    print(f"nn_logmag_windows_shape={tuple(nn_logmag_windows.shape)}")


if __name__ == "__main__":
    main()
