"""Tests for live dashboard rendering helpers."""

from rich.panel import Panel

from mmwave_radar_devtool.cfg_parser import parse_radar_cfg
from mmwave_radar_devtool.config import CaptureConfig
from mmwave_radar_devtool.live_view import (
    PlotSeries,
    SignalViewMode,
    TerminalLiveDashboard,
    _render_line_plot,
)


def test_capture_config_allows_unbounded_live_runs() -> None:
    """Live capture config should permit no duration."""
    config = CaptureConfig(duration_s=None)
    assert config.duration_s is None


def test_render_line_plot_returns_panel() -> None:
    """Plot rendering should produce a Rich panel."""
    panel = _render_line_plot(
        PlotSeries(
            values=[1.0, 2.0, 3.0],
            title="Test",
            left_label="0",
            right_label="3",
            unit="u",
            accent="#22d3ee",
        ),
        width=40,
        height=12,
    )
    assert isinstance(panel, Panel)


def test_dashboard_builds_renderable() -> None:
    """The dashboard should build a renderable layout."""
    dashboard = TerminalLiveDashboard()
    renderable = dashboard._render()
    assert renderable is not None


def test_dashboard_can_switch_modes_with_cfg_context() -> None:
    """The dashboard should expose different post-processing modes."""
    cfg = parse_radar_cfg("config/xwr18xx_profile_raw_capture.cfg")
    dashboard = TerminalLiveDashboard(radar_cfg=cfg)
    dashboard.metrics.current_mode = SignalViewMode.RANGE
    plot_series = dashboard._build_plot_series()
    assert plot_series.title.startswith("Approximate range profile")
