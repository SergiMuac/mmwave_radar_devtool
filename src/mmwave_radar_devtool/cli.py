"""Command-line interface for the package."""

from __future__ import annotations

import argparse
from pathlib import Path

from .capture import CaptureOrchestrator
from .cfg_parser import create_capture_cfg, parse_radar_cfg
from .config import CaptureConfig, DCA1000Config, RadarSerialConfig
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
    orchestrator, cfg = _build_runtime(args)
    output_path = Path(args.output) if args.output else None
    stats = orchestrator.capture_live(
        cfg=cfg,
        capture_config=CaptureConfig(
            output_path=output_path,
            duration_s=args.duration,
            strip_dca_header=not args.keep_dca_header,
        ),
    )
    _print_stats(stats)
    return 0


def _handle_plot(args: argparse.Namespace) -> int:
    """Handle the plot command."""
    plot_raw_iq(path=args.input, sample_count=args.samples)
    return 0


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


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
