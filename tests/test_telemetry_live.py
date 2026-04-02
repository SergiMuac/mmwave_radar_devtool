"""Tests for TI telemetry live dashboard rendering."""

from mmwave_radar_devtool.cfg_parser import parse_radar_cfg
from mmwave_radar_devtool.telemetry_live import TerminalTelemetryDashboard
from mmwave_radar_devtool.usb_telemetry import TiTelemetryFrame


def test_dashboard_prefers_zero_doppler_range_profile_plot() -> None:
    """Telemetry live view should prioritize the TI range-profile TLV."""
    dashboard = TerminalTelemetryDashboard(
        radar_cfg=parse_radar_cfg("config/xwr68xx_profile_2026_04_02T12_19_57_603.cfg")
    )
    dashboard.metrics.record_frame(
        TiTelemetryFrame(
            version=0,
            total_packet_length=0,
            platform=0,
            frame_number=1,
            time_cpu_cycles=0,
            num_detected_objects=0,
            num_tlvs=1,
            subframe_number=0,
            tlvs=(),
            points=(),
            range_profile=(1.0, 2.0, 3.0),
        )
    )

    series = dashboard._build_plot_series()

    assert series.title == "Zero-Doppler range profile"
    assert series.right_label.endswith("m")
