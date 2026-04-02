"""Live terminal dashboard for TI USB/UART telemetry streams."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .cfg_parser import RadarCliConfig
from .live_view import C_M_PER_S, PlotSeries, _metric_tile, _render_line_plot
from .usb_telemetry import TiTelemetryFrame


@dataclass(slots=True)
class TelemetryLiveMetrics:
    """Mutable live statistics for TI telemetry mode."""

    started_at: float = field(default_factory=time.monotonic)
    frames_received: int = 0
    bytes_received: int = 0
    tlvs_received: int = 0
    last_frame_number: int | None = None
    latest_detected_points: int = 0
    latest_range_profile: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    latest_ranges_m: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    latest_snr_db: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def record_frame(self, frame: TiTelemetryFrame) -> None:
        """Update live metrics from one telemetry frame."""
        self.frames_received += 1
        self.bytes_received += frame.total_packet_length
        self.tlvs_received += len(frame.tlvs)
        self.last_frame_number = frame.frame_number
        self.latest_detected_points = frame.detected_points
        self.latest_range_profile = np.array(frame.range_profile, dtype=np.float32)
        self.latest_ranges_m = np.array([point.range_m for point in frame.points], dtype=np.float32)
        self.latest_snr_db = np.array(
            [point.snr_db for point in frame.points if point.snr_db is not None],
            dtype=np.float32,
        )


class TerminalTelemetryDashboard:
    """Rich terminal dashboard for TI USB telemetry."""

    def __init__(
        self,
        title: str = "TI USB Telemetry Live",
        radar_cfg: RadarCliConfig | None = None,
    ) -> None:
        """Initialize the telemetry dashboard."""
        self._title = title
        self._metrics = TelemetryLiveMetrics()
        self._console = Console()
        self._live: Live | None = None
        self._active = False
        self._stop_requested = False
        self._profile_cfg = radar_cfg.parse_profile_cfg() if radar_cfg is not None else None

    @property
    def metrics(self) -> TelemetryLiveMetrics:
        """Expose current metrics."""
        return self._metrics

    @property
    def stop_requested(self) -> bool:
        """Return whether the live session should stop."""
        return self._stop_requested

    def start(self) -> None:
        """Enter dashboard mode."""
        if not self._console.is_terminal or self._active:
            return
        self._live = Live(self._render(), console=self._console, auto_refresh=True, screen=True)
        self._live.start()
        self._active = True

    def stop(self) -> None:
        """Leave dashboard mode."""
        if not self._active or self._live is None:
            return
        self._live.stop()
        self._live = None
        self._active = False

    def update(self) -> None:
        """Render the current dashboard state."""
        if not self._active or self._live is None:
            return
        self._live.update(self._render(), refresh=True)

    def _render(self) -> RenderableType:
        """Build the dashboard frame."""
        width, height = shutil.get_terminal_size(fallback=(140, 42))
        runtime_s = time.monotonic() - self._metrics.started_at
        summary = Table.grid(expand=True)
        for _ in range(4):
            summary.add_column(ratio=1)
        summary.add_row(
            _metric_tile("Runtime", f"{runtime_s:,.1f}s", "#22c55e"),
            _metric_tile("Frames", f"{self._metrics.frames_received:,}", "#06b6d4"),
            _metric_tile("TLVs", f"{self._metrics.tlvs_received:,}", "#8b5cf6"),
            _metric_tile("Points", f"{self._metrics.latest_detected_points:,}", "#f59e0b"),
        )
        summary.add_row(
            _metric_tile("Bytes", f"{self._metrics.bytes_received:,}", "#14b8a6"),
            _metric_tile("Frame No", str(self._metrics.last_frame_number), "#3b82f6"),
            _metric_tile("Range Bins", str(self._metrics.latest_range_profile.size), "#a855f7"),
            _metric_tile("SNR Bins", str(self._metrics.latest_snr_db.size), "#ef4444"),
        )

        plot = _render_line_plot(
            self._build_plot_series(),
            width=max(40, width - 10),
            height=max(14, height - 18),
        )

        layout = Table.grid(expand=True)
        layout.add_column(ratio=1)
        layout.add_row(Panel(summary, title=self._title, border_style="#0f172a", padding=(0, 1)))
        layout.add_row(plot)
        return layout

    def _build_plot_series(self) -> PlotSeries:
        """Render zero-Doppler range profile or point-derived fallback values."""
        if self._metrics.latest_range_profile.size > 0:
            return PlotSeries(
                values=self._metrics.latest_range_profile.astype(np.float32),
                title="Zero-Doppler range profile",
                left_label="0 m",
                right_label=self._format_max_range(self._metrics.latest_range_profile.size),
                unit="mag",
                accent="#22d3ee",
            )
        if self._metrics.latest_ranges_m.size > 0:
            values = np.sort(self._metrics.latest_ranges_m).astype(np.float32)
            return PlotSeries(
                values=values,
                title="Detected point ranges",
                left_label="nearest",
                right_label="farthest",
                unit="m",
                accent="#22d3ee",
            )
        if self._metrics.latest_snr_db.size > 0:
            values = self._metrics.latest_snr_db.astype(np.float32)
            return PlotSeries(
                values=values,
                title="Detected point SNR",
                left_label="first",
                right_label="last",
                unit="dB",
                accent="#34d399",
            )
        return PlotSeries(
            values=np.zeros(128, dtype=np.float32),
            title="Detected point ranges",
            left_label="no detections",
            right_label="awaiting frame",
            unit="m",
            accent="#22d3ee",
        )

    def _format_max_range(self, bin_count: int) -> str:
        """Format the right-edge range axis label from profileCfg parameters."""
        profile = self._profile_cfg
        if profile is None or bin_count <= 0:
            return "range"
        sample_rate_hz = profile.dig_out_sample_rate_ksps * 1_000.0
        slope_hz_per_s = profile.freq_slope_mhz_per_us * 1_000_000_000_000.0
        max_beat_hz = sample_rate_hz / 2.0
        max_range_m = C_M_PER_S * max_beat_hz / (2.0 * slope_hz_per_s)
        return f"{max_range_m:.2f} m"
