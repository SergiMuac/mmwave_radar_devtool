"""Command-line interface for the package."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .capture import CaptureOrchestrator
from .cfg_parser import create_capture_cfg, parse_radar_cfg
from .config import CaptureConfig, DCA1000Config, RadarSerialConfig
from .live_view import LivePredictionResult, LiveSignalProcessingConfig
from .range_profile_dataset import (
    decode_complex_xwr18xx,
    load_raw_words,
    make_range_window,
    parse_tensor_cfg,
    select_useful_range_side,
)
from .visualize import plot_raw_iq


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="mmw")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--radar-cli-port", required=True)
    common.add_argument("--cfg", required=True)
    common.add_argument("--fpga-ip", default="192.168.33.180")
    common.add_argument("--host-ip", default="192.168.33.30")
    common.add_argument("--config-port", type=int, default=4096)
    common.add_argument("--data-port", type=int, default=4098)
    common.add_argument("--baudrate", type=int, default=115200)
    common.add_argument("--debug-serial", action="store_true")
    common.add_argument("--verbose", action="store_true")
    common.add_argument("--packet-delay-us", type=int, default=10)
    common.add_argument("--fpga-config-timer-s", type=int, default=30)

    probe_parser = subparsers.add_parser("probe", parents=[common])
    probe_parser.set_defaults(handler=_handle_probe)

    capture_parser = subparsers.add_parser("capture", parents=[common])
    capture_parser.add_argument("--output", required=True)
    capture_parser.add_argument("--duration", type=float, required=True)
    capture_parser.add_argument("--keep-dca-header", action="store_true")
    capture_parser.set_defaults(handler=_handle_capture)

    live_parser = subparsers.add_parser("live", parents=[common])
    live_parser.add_argument("--duration", type=float)
    live_parser.add_argument("--output")
    live_parser.add_argument("--keep-dca-header", action="store_true")
    live_parser.add_argument(
        "--live-baseline-capture",
        help="Optional baseline .bin used for live range-delta visualization.",
    )
    live_parser.add_argument("--live-decode-order", choices=["iiqq", "iqiq"], default="iiqq")
    live_parser.add_argument("--live-window-kind", choices=["hann", "rect"], default="hann")
    live_parser.add_argument(
        "--live-range-side",
        choices=["positive", "negative", "full"],
        default="positive",
    )
    live_parser.add_argument("--live-spectrum-db-min", type=float, default=0.0)
    live_parser.add_argument("--live-spectrum-db-max", type=float, default=120.0)
    live_parser.add_argument("--live-range-db-min", type=float, default=0.0)
    live_parser.add_argument("--live-range-db-max", type=float, default=120.0)
    live_parser.add_argument(
        "--live-chirp-align-offset",
        type=int,
        default=-1,
        help="Chirp alignment in complex samples; set -1 for auto estimation.",
    )
    live_parser.add_argument("--live-normalize-spectrum", action="store_true")
    live_parser.add_argument("--live-normalize-range", action="store_true")
    live_parser.add_argument(
        "--live-ml-checkpoint",
        type=Path,
        help="Optional trained classifier/regressor checkpoint for live NN predictions.",
    )
    live_parser.add_argument(
        "--live-ml-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device for --live-ml-checkpoint.",
    )
    live_parser.add_argument(
        "--live-ml-interval-s",
        type=float,
        default=0.5,
        help="Minimum seconds between live NN predictions.",
    )
    live_parser.add_argument(
        "--live-ml-window-frames",
        type=int,
        default=16,
        help="Rolling frame window used for live NN prediction aggregation.",
    )
    live_parser.set_defaults(handler=_handle_live)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--input", required=True)
    plot_parser.add_argument("--samples", type=int, default=4096)
    plot_parser.set_defaults(handler=_handle_plot)

    make_cfg_parser = subparsers.add_parser("make-capture-cfg")
    make_cfg_parser.add_argument("--input", required=True)
    make_cfg_parser.add_argument("--output", required=True)
    make_cfg_parser.add_argument("--enable-header", action="store_true")
    make_cfg_parser.add_argument("--disable-hw-stream", action="store_true")
    make_cfg_parser.add_argument("--enable-sw-stream", action="store_true")
    make_cfg_parser.set_defaults(handler=_handle_make_capture_cfg)

    return parser


def _build_runtime(args: argparse.Namespace) -> tuple[CaptureOrchestrator, object]:
    """Build shared runtime objects for CLI commands."""
    dca_config = DCA1000Config(
        fpga_ip=args.fpga_ip,
        host_ip=args.host_ip,
        config_port=args.config_port,
        data_port=args.data_port,
        packet_delay_us=args.packet_delay_us,
        fpga_config_timer_s=args.fpga_config_timer_s,
    )
    serial_config = RadarSerialConfig(
        cli_port=args.radar_cli_port,
        cli_baudrate=args.baudrate,
        debug_serial=args.debug_serial,
        verbose=args.verbose,
    )
    orchestrator = CaptureOrchestrator(dca_config=dca_config, serial_config=serial_config)
    cfg = parse_radar_cfg(Path(args.cfg))
    return orchestrator, cfg


def _handle_probe(args: argparse.Namespace) -> int:
    """Handle the probe command."""
    orchestrator, cfg = _build_runtime(args)
    result = orchestrator.probe(cfg=cfg)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


def _handle_capture(args: argparse.Namespace) -> int:
    """Handle the capture command."""
    orchestrator, cfg = _build_runtime(args)
    stats = orchestrator.capture(
        cfg=cfg,
        capture_config=CaptureConfig(
            output_path=Path(args.output),
            duration_s=args.duration,
            strip_dca_header=not args.keep_dca_header,
        ),
    )
    _print_stats(stats)
    return 0


def _handle_live(args: argparse.Namespace) -> int:
    """Handle the live terminal viewer command."""
    if args.live_ml_window_frames <= 0:
        raise ValueError("--live-ml-window-frames must be > 0")
    orchestrator, cfg = _build_runtime(args)
    output_path = Path(args.output) if args.output else None
    baseline_range_db = None
    if args.live_baseline_capture:
        baseline_range_db = _load_live_baseline_range_db(
            baseline_capture=Path(args.live_baseline_capture),
            cfg_path=Path(args.cfg),
            window_kind=args.live_window_kind,
            range_side=args.live_range_side,
        )
    range_db_min = args.live_range_db_min
    range_db_max = args.live_range_db_max
    if baseline_range_db is None and range_db_min is None:
        range_db_min = 0.0
    elif baseline_range_db is not None:
        if range_db_min is None:
            range_db_min = 0.0
        if range_db_max is None:
            range_db_max = 120.0
    prediction_callback = None
    if args.live_ml_checkpoint:
        from .ml.predict import load_radar_predictor

        predictor = load_radar_predictor(
            Path(args.live_ml_checkpoint),
            cfg_path=Path(args.cfg),
            device_name=args.live_ml_device,
        )
        if predictor.feature_mode != "zero_doppler_db":
            raise ValueError(
                "Live NN prediction currently supports feature_mode='zero_doppler_db' only. "
                f"Loaded checkpoint feature_mode={predictor.feature_mode!r}."
            )

        def prediction_callback(frame_db: np.ndarray) -> LivePredictionResult:
            summary = predictor.predict_frame_db(
                frame_db,
                live_window_frames=int(args.live_ml_window_frames),
            )
            return LivePredictionResult(
                task=summary.task,
                primary=summary.primary,
                confidence=summary.confidence,
                detail=summary.detail,
            )

        print(
            "Live NN prediction enabled "
            f"(task={predictor.task}, checkpoint={Path(args.live_ml_checkpoint)})"
        )
    live_processing = LiveSignalProcessingConfig(
        decode_order=args.live_decode_order,
        range_window_kind=args.live_window_kind,
        range_side=args.live_range_side,
        normalize_spectrum_to_peak=args.live_normalize_spectrum,
        normalize_range_to_peak=args.live_normalize_range,
        spectrum_db_min=args.live_spectrum_db_min,
        spectrum_db_max=args.live_spectrum_db_max,
        range_db_min=range_db_min,
        range_db_max=range_db_max,
        chirp_alignment_offset=args.live_chirp_align_offset,
        baseline_range_db=baseline_range_db,
        prediction_callback=prediction_callback,
        prediction_interval_s=float(args.live_ml_interval_s),
    )
    stats = orchestrator.capture_live(
        cfg=cfg,
        capture_config=CaptureConfig(
            output_path=output_path,
            duration_s=args.duration,
            strip_dca_header=not args.keep_dca_header,
        ),
        live_processing_config=live_processing,
    )
    _print_stats(stats)
    return 0


def _handle_plot(args: argparse.Namespace) -> int:
    """Handle the plot command."""
    plot_raw_iq(path=args.input, sample_count=args.samples)
    return 0


def _load_live_baseline_range_db(
    *,
    baseline_capture: Path,
    cfg_path: Path,
    window_kind: str,
    range_side: str,
    eps: float = 1e-9,
) -> np.ndarray:
    """Load baseline and return mean zero-Doppler profile in dB for live subtraction."""
    tensor_cfg = parse_tensor_cfg(cfg_path)
    words = load_raw_words(baseline_capture)
    usable_words = (words.size // 4) * 4
    if usable_words <= 0:
        raise ValueError(
            "Baseline capture does not contain enough I/Q words "
            f"for decode: {baseline_capture}"
        )
    if usable_words < words.size:
        print(
            "Warning: baseline capture has trailing partial I/Q group; "
            f"dropping {words.size - usable_words} int16 words."
        )
    complex_stream = decode_complex_xwr18xx(words[:usable_words])

    samples_per_frame = tensor_cfg.complex_samples_per_frame
    usable_complex = (complex_stream.size // samples_per_frame) * samples_per_frame
    if usable_complex <= 0:
        raise ValueError(
            "Baseline capture does not include one full frame after trimming. "
            f"required_complex_per_frame={samples_per_frame} "
            f"available_complex={complex_stream.size} "
            f"capture={baseline_capture}"
        )
    if usable_complex < complex_stream.size:
        print(
            "Warning: baseline capture has trailing partial frame; "
            f"dropping {complex_stream.size - usable_complex} complex samples."
        )

    cube = complex_stream[:usable_complex].reshape(
        -1,
        tensor_cfg.chirps_per_frame,
        tensor_cfg.num_rx,
        tensor_cfg.num_adc_samples,
    )
    window = make_range_window(tensor_cfg.num_adc_samples, kind=window_kind)
    range_cube = np.fft.fft(cube * window[None, None, None, :], axis=-1)
    doppler_cube = np.fft.fft(range_cube, axis=1)
    power = np.mean(np.abs(doppler_cube[:, 0, :, :]) ** 2, axis=1).astype(np.float32)

    if range_side != "full":
        power = select_useful_range_side(power, side=range_side)
    if power.size == 0:
        raise ValueError(
            "Baseline capture produced no usable zero-Doppler bins: "
            f"{baseline_capture}"
        )
    mean_power = np.mean(power, axis=0, dtype=np.float32)
    return (10.0 * np.log10(mean_power + float(eps))).astype(np.float32)


def _handle_make_capture_cfg(args: argparse.Namespace) -> int:
    """Create a capture-ready cfg from an existing cfg."""
    output_path = create_capture_cfg(
        source_path=Path(args.input),
        target_path=Path(args.output),
        enable_header=args.enable_header,
        enable_hw_stream=not args.disable_hw_stream,
        enable_sw_stream=args.enable_sw_stream,
    )
    print(output_path)
    return 0


def _print_stats(stats: object) -> None:
    """Print capture statistics."""
    print(f"packets_received={stats.packets_received}")
    print(f"bytes_received={stats.bytes_received}")
    print(f"payload_bytes_written={stats.payload_bytes_written}")
    print(f"elapsed_s={stats.elapsed_s:.3f}")
    print(f"first_sequence_number={stats.first_sequence_number}")
    print(f"last_sequence_number={stats.last_sequence_number}")
    print(f"sequence_gaps_detected={stats.sequence_gaps_detected}")
    print(f"malformed_datagrams_detected={stats.malformed_datagrams_detected}")


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
