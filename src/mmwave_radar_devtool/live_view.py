"""Rich live dashboard for radar capture telemetry and simple signal views."""

from __future__ import annotations

import math
import shutil
import sys
import termios
import threading
import time
import tty
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .cfg_parser import ProfileCfg, RadarCliConfig
from .dca1000 import DCA1000DataPacket

C_M_PER_S = 299_792_458.0


class SignalViewMode(str, Enum):
    """Selectable signal processing modes for the live plot."""

    RAW = "raw"
    INTENSITY = "intensity"
    SPECTRUM = "spectrum"
    RANGE = "range"


@dataclass(slots=True, frozen=True)
class PlotSeries:
    """A renderable one-dimensional signal series."""

    values: np.ndarray
    title: str
    left_label: str
    right_label: str
    unit: str
    accent: str


@dataclass(slots=True)
class LiveMetrics:
    """Mutable live statistics for the terminal dashboard."""

    started_at: float = field(default_factory=time.monotonic)
    packets_received: int = 0
    bytes_received: int = 0
    payload_bytes_received: int = 0
    first_sequence_number: int | None = None
    last_sequence_number: int | None = None
    sequence_gaps_detected: int = 0
    recent_packet_rates: deque[float] = field(default_factory=lambda: deque(maxlen=180))
    recent_throughput_rates: deque[float] = field(default_factory=lambda: deque(maxlen=180))
    packets_since_last_rate: int = 0
    bytes_since_last_rate: int = 0
    last_rate_timestamp: float = field(default_factory=time.monotonic)
    current_mode: SignalViewMode = SignalViewMode.RAW
    latest_raw: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    latest_complex: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.complex64))

    def record_packet(self, packet: DCA1000DataPacket) -> None:
        """Update live metrics from a packet."""
        self.packets_received += 1
        self.bytes_received += packet.byte_count + 10
        self.payload_bytes_received += len(packet.payload)

        if self.first_sequence_number is None:
            self.first_sequence_number = packet.sequence_number
        elif (
            self.last_sequence_number is not None
            and packet.sequence_number != self.last_sequence_number + 1
        ):
            self.sequence_gaps_detected += max(
                0, packet.sequence_number - self.last_sequence_number - 1
            )

        self.last_sequence_number = packet.sequence_number
        self.packets_since_last_rate += 1
        self.bytes_since_last_rate += len(packet.payload)
        self._update_signal_buffers(packet.payload)
        self._update_rates()

    def _update_rates(self) -> None:
        """Update rate windows if enough time elapsed."""
        now = time.monotonic()
        elapsed = now - self.last_rate_timestamp
        if elapsed < 0.08:
            return
        self.recent_packet_rates.append(self.packets_since_last_rate / elapsed)
        self.recent_throughput_rates.append(self.bytes_since_last_rate / elapsed)
        self.packets_since_last_rate = 0
        self.bytes_since_last_rate = 0
        self.last_rate_timestamp = now

    def _update_signal_buffers(self, payload: bytes) -> None:
        """Decode raw payload bytes into real and complex sample views."""
        if not payload:
            return
        sample_count = len(payload) // 2
        if sample_count <= 0:
            return
        samples = np.frombuffer(payload[: sample_count * 2], dtype="<i2").astype(np.float32)
        if samples.size < 4:
            return
        self.latest_raw = samples[-2048:]
        even_count = (samples.size // 2) * 2
        iq = samples[:even_count].reshape(-1, 2)
        self.latest_complex = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)[-1024:]


@dataclass(slots=True)
class InputController:
    """Background keyboard controller for mode selection."""

    mode_callback: Callable[[SignalViewMode], None]
    stop_callback: Callable[[], None]
    _thread: threading.Thread | None = None
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _original_termios: list[int] | None = None

    def start(self) -> None:
        """Start non-blocking key capture when stdin is a terminal."""
        if not sys.stdin.isatty():
            return
        try:
            self._original_termios = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        except termios.error:
            self._original_termios = None
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop key capture and restore terminal settings."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        if self._original_termios is not None and sys.stdin.isatty():
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._original_termios)
            except termios.error:
                pass
            self._original_termios = None

    def _run(self) -> None:
        """Listen for mode-selection keys."""
        import select

        while not self._stop_event.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue
            key = sys.stdin.read(1)
            if key == "1":
                self.mode_callback(SignalViewMode.RAW)
            elif key == "2":
                self.mode_callback(SignalViewMode.INTENSITY)
            elif key == "3":
                self.mode_callback(SignalViewMode.SPECTRUM)
            elif key == "4":
                self.mode_callback(SignalViewMode.RANGE)
            elif key in {"q", "Q"}:
                self.stop_callback()
            elif key == "\x03":
                self.stop_callback()


class TerminalLiveDashboard:
    """Rich terminal dashboard for live radar telemetry and simple plots."""

    def __init__(
        self, radar_cfg: RadarCliConfig | None = None, title: str = "TI mmWave + DCA1000 Live"
    ) -> None:
        """Initialize the dashboard."""
        self._title = title
        self._metrics = LiveMetrics()
        self._console = Console()
        self._live: Live | None = None
        self._active = False
        self._stop_requested = False
        self._profile_cfg = radar_cfg.parse_profile_cfg() if radar_cfg is not None else None
        self._input = InputController(
            mode_callback=self._set_mode,
            stop_callback=self.request_stop,
        )

    @property
    def metrics(self) -> LiveMetrics:
        """Expose current metrics."""
        return self._metrics

    @property
    def stop_requested(self) -> bool:
        """Return whether the live session should stop."""
        return self._stop_requested

    def request_stop(self) -> None:
        """Request termination of the live session."""
        self._stop_requested = True

    def start(self) -> None:
        """Enter dashboard mode."""
        if not self._console.is_terminal or self._active:
            return
        self._live = Live(
            self._render(),
            console=self._console,
            auto_refresh=False,
            screen=True,
            transient=False,
        )
        self._live.start()
        self._input.start()
        self._active = True

    def stop(self) -> None:
        """Leave dashboard mode."""
        self._input.stop()
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

    def _set_mode(self, mode: SignalViewMode) -> None:
        """Switch the primary plot mode."""
        self._metrics.current_mode = mode
        self.update()

    def _render(self) -> RenderableType:
        """Build the dashboard frame."""
        width, height = shutil.get_terminal_size(fallback=(140, 42))
        runtime_s = time.monotonic() - self._metrics.started_at
        packet_rate = (
            self._metrics.recent_packet_rates[-1] if self._metrics.recent_packet_rates else 0.0
        )
        throughput = (
            self._metrics.recent_throughput_rates[-1]
            if self._metrics.recent_throughput_rates
            else 0.0
        )
        main_plot = _render_line_plot(
            self._build_plot_series(),
            width=max(40, width - 10),
            height=max(14, height - 18),
        )

        summary = Table.grid(expand=True)
        for _ in range(4):
            summary.add_column(ratio=1)
        summary.add_row(
            _metric_tile("Runtime", f"{runtime_s:,.1f}s", "#22c55e"),
            _metric_tile("Packets", f"{self._metrics.packets_received:,}", "#06b6d4"),
            _metric_tile("Throughput", _format_bytes_per_second(throughput), "#8b5cf6"),
            _metric_tile("Rate", f"{packet_rate:,.1f} pkt/s", "#f59e0b"),
        )
        summary.add_row(
            _metric_tile("Payload", _format_bytes(self._metrics.payload_bytes_received), "#14b8a6"),
            _metric_tile("First seq", str(self._metrics.first_sequence_number), "#3b82f6"),
            _metric_tile("Last seq", str(self._metrics.last_sequence_number), "#a855f7"),
            _metric_tile("Gaps", f"{self._metrics.sequence_gaps_detected:,}", "#ef4444"),
        )

        controls = Panel(
            _render_mode_tabs(self._metrics.current_mode),
            title="Views",
            subtitle="Keys: 1 Raw   2 Intensity   3 Spectrum   4 Range FFT   q Quit",
            border_style="#334155",
            padding=(0, 1),
        )

        layout = Table.grid(expand=True)
        layout.add_column(ratio=1)
        layout.add_row(Panel(summary, title=self._title, border_style="#0f172a", padding=(0, 1)))
        layout.add_row(controls)
        layout.add_row(main_plot)
        return layout

    def _build_plot_series(self) -> PlotSeries:
        """Create the currently selected plot series."""
        if self._metrics.current_mode is SignalViewMode.RAW:
            return self._build_raw_series()
        if self._metrics.current_mode is SignalViewMode.INTENSITY:
            return self._build_intensity_series()
        if self._metrics.current_mode is SignalViewMode.SPECTRUM:
            return self._build_spectrum_series()
        return self._build_range_series()

    def _build_raw_series(self) -> PlotSeries:
        """Build a raw sample waveform plot."""
        values = self._metrics.latest_raw
        if values.size == 0:
            values = np.zeros(128, dtype=np.float32)
        return PlotSeries(
            values=values.astype(np.float32),
            title="Raw ADC waveform",
            left_label="0",
            right_label=f"{values.size} samples",
            unit="adc",
            accent="#22d3ee",
        )

    def _build_intensity_series(self) -> PlotSeries:
        """Build a short-window intensity envelope plot."""
        values = np.abs(self._metrics.latest_complex)
        if values.size == 0:
            values = np.zeros(128, dtype=np.float32)
        if values.size >= 8:
            kernel = np.ones(8, dtype=np.float32) / 8.0
            values = np.convolve(values.astype(np.float32), kernel, mode="same")
        return PlotSeries(
            values=values.astype(np.float32),
            title="Instantaneous intensity envelope",
            left_label="near",
            right_label="farther samples",
            unit="|IQ|",
            accent="#34d399",
        )

    def _build_spectrum_series(self) -> PlotSeries:
        """Build a frequency-domain magnitude spectrum."""
        values = self._metrics.latest_complex
        if values.size == 0:
            spectrum = np.zeros(128, dtype=np.float32)
        else:
            windowed = values * np.hanning(values.size)
            spectrum = np.abs(np.fft.fft(windowed))[: max(8, values.size // 2)]
        return PlotSeries(
            values=spectrum.astype(np.float32),
            title="Beat-frequency magnitude spectrum",
            left_label="0 Hz",
            right_label=self._format_max_frequency(len(spectrum)),
            unit="mag",
            accent="#c084fc",
        )

    def _build_range_series(self) -> PlotSeries:
        """Build a simple range FFT magnitude plot."""
        values = self._metrics.latest_complex
        if values.size == 0:
            spectrum = np.zeros(128, dtype=np.float32)
        else:
            windowed = values * np.hanning(values.size)
            spectrum = np.abs(np.fft.fft(windowed))[: max(8, values.size // 2)]
        return PlotSeries(
            values=spectrum.astype(np.float32),
            title="Approximate range profile (FFT)",
            left_label="0 m",
            right_label=self._format_max_range(len(spectrum)),
            unit="mag",
            accent="#f59e0b",
        )

    def _format_max_frequency(self, bin_count: int) -> str:
        """Format the right-edge frequency axis label."""
        profile = self._profile_cfg
        if profile is None:
            return "Nyquist"
        sample_rate_hz = profile.dig_out_sample_rate_ksps * 1_000.0
        max_freq_hz = sample_rate_hz / 2.0
        if max_freq_hz >= 1_000_000.0:
            return f"{max_freq_hz / 1_000_000.0:.2f} MHz"
        if max_freq_hz >= 1_000.0:
            return f"{max_freq_hz / 1_000.0:.1f} kHz"
        return f"{max_freq_hz:.0f} Hz"

    def _format_max_range(self, bin_count: int) -> str:
        """Format the right-edge range axis label."""
        profile = self._profile_cfg
        if profile is None or bin_count <= 0:
            return "range"
        sample_rate_hz = profile.dig_out_sample_rate_ksps * 1_000.0
        slope_hz_per_s = profile.freq_slope_mhz_per_us * 1_000_000_000_000.0
        max_beat_hz = sample_rate_hz / 2.0
        max_range_m = C_M_PER_S * max_beat_hz / (2.0 * slope_hz_per_s)
        return f"{max_range_m:.2f} m"


def _format_bytes(value: float) -> str:
    """Format a byte quantity using binary units."""
    units = ("B", "KiB", "MiB", "GiB")
    scaled = value
    unit = units[0]
    for unit in units:
        if scaled < 1024.0 or unit == units[-1]:
            break
        scaled /= 1024.0
    return f"{scaled:.2f} {unit}"


def _format_bytes_per_second(value: float) -> str:
    """Format a throughput value."""
    return f"{_format_bytes(value)}/s"


def _metric_tile(title: str, value: str, accent: str) -> Panel:
    """Render a compact metric tile."""
    body = Table.grid(expand=True)
    body.add_column(justify="center")
    body.add_row(Text(title, style="bold #94a3b8"))
    body.add_row(Text(value, style=f"bold {accent}"))
    return Panel(body, border_style=accent, padding=(0, 1))


def _render_mode_tabs(current_mode: SignalViewMode) -> Text:
    """Render the mode selector as terminal tabs."""
    tabs = Text()
    items = [
        ("1", SignalViewMode.RAW, "Raw"),
        ("2", SignalViewMode.INTENSITY, "Intensity"),
        ("3", SignalViewMode.SPECTRUM, "Spectrum"),
        ("4", SignalViewMode.RANGE, "Range FFT"),
    ]
    for key, mode, label in items:
        if mode is current_mode:
            tabs.append(f" {key}:{label} ", style="bold black on #38bdf8")
        else:
            tabs.append(f" {key}:{label} ", style="bold #cbd5e1 on #1e293b")
        tabs.append("  ")
    return tabs


def _resample_to_width(values: np.ndarray, width: int) -> np.ndarray:
    """Resample a one-dimensional series to an exact width."""
    if width <= 0:
        return np.zeros(0, dtype=np.float32)
    source = np.asarray(values, dtype=np.float32)
    if source.size == 0:
        return np.zeros(width, dtype=np.float32)
    if source.size == 1:
        return np.full(width, float(source[0]), dtype=np.float32)
    positions = np.linspace(0, source.size - 1, width, dtype=np.float32)
    return np.interp(positions, np.arange(source.size, dtype=np.float32), source).astype(np.float32)


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a light smoothing filter for more continuous terminal plots."""
    series = np.asarray(values, dtype=np.float32)
    if series.size < 3 or window <= 1:
        return series
    kernel_width = min(window, max(3, series.size // 8))
    if kernel_width % 2 == 0:
        kernel_width += 1
    kernel = np.ones(kernel_width, dtype=np.float32) / float(kernel_width)
    return np.convolve(series, kernel, mode="same").astype(np.float32)


def _render_line_plot(series: PlotSeries, *, width: int, height: int) -> Panel:
    """Render a smooth high-resolution line plot using Braille cells."""
    plot_width = max(16, width - 12)
    plot_height = max(8, height - 4)
    subpixel_width = plot_width * 2
    subpixel_height = plot_height * 4

    base_values = np.asarray(series.values, dtype=np.float32)
    values = _resample_to_width(base_values, subpixel_width)
    values = _smooth_series(values, window=max(5, subpixel_width // 64))

    minimum = float(np.min(values)) if values.size else 0.0
    maximum = float(np.max(values)) if values.size else 0.0
    span = maximum - minimum
    if span <= 1e-9:
        normalized = np.full(values.shape, 0.5, dtype=np.float32)
    else:
        normalized = (values - minimum) / span

    bitmap = [[False for _ in range(subpixel_width)] for _ in range(subpixel_height)]
    points: list[tuple[int, int]] = []
    for index, fraction in enumerate(normalized):
        x = min(subpixel_width - 1, index)
        y = min(
            subpixel_height - 1,
            max(0, int(round((1.0 - float(fraction)) * (subpixel_height - 1)))),
        )
        points.append((x, y))

    for start, end in zip(points, points[1:]):
        _draw_bitmap_line(bitmap, start[0], start[1], end[0], end[1])

    rendered_lines: list[Text] = []
    for row in range(plot_height):
        axis_fraction = 1.0 - (row / max(1, plot_height - 1))
        axis_value = minimum + axis_fraction * (span if span > 1e-9 else 1.0)
        line = Text(f"{axis_value:>8.1f} │", style="#64748b")
        for col in range(plot_width):
            character = _braille_cell(bitmap, col * 2, row * 4)
            if character == " ":
                line.append(" ")
            else:
                line.append(character, style=f"bold {series.accent}")
        rendered_lines.append(line)

    footer = Text(f"{series.left_label:<12}", style="#64748b")
    footer.append(
        "─" * max(1, plot_width - len(series.left_label) - len(series.right_label) - 2),
        style="#334155",
    )
    footer.append(f"{series.right_label:>12}", style="#64748b")
    rendered_lines.append(footer)
    rendered_lines.append(Text(f"Unit: {series.unit}", style="dim"))

    return Panel(
        Group(*rendered_lines),
        title=series.title,
        border_style=series.accent,
        padding=(0, 1),
    )


def _draw_bitmap_line(bitmap: list[list[bool]], x0: int, y0: int, x1: int, y1: int) -> None:
    """Rasterize a line on a high-resolution boolean bitmap."""
    width = len(bitmap[0]) if bitmap else 0
    height = len(bitmap)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            bitmap[y0][x0] = True
        if x0 == x1 and y0 == y1:
            break
        double_error = 2 * error
        if double_error >= dy:
            error += dy
            x0 += step_x
        if double_error <= dx:
            error += dx
            y0 += step_y


def _braille_cell(bitmap: list[list[bool]], origin_x: int, origin_y: int) -> str:
    """Convert a 2x4 bitmap tile into a Unicode Braille character."""
    dot_offsets = (
        (0, 0, 0x01),
        (0, 1, 0x02),
        (0, 2, 0x04),
        (1, 0, 0x08),
        (1, 1, 0x10),
        (1, 2, 0x20),
        (0, 3, 0x40),
        (1, 3, 0x80),
    )
    codepoint = 0
    height = len(bitmap)
    width = len(bitmap[0]) if bitmap else 0
    for offset_x, offset_y, bit in dot_offsets:
        x = origin_x + offset_x
        y = origin_y + offset_y
        if 0 <= x < width and 0 <= y < height and bitmap[y][x]:
            codepoint |= bit
    if codepoint == 0:
        return " "
    return chr(0x2800 + codepoint)


__all__ = [
    "LiveMetrics",
    "PlotSeries",
    "SignalViewMode",
    "TerminalLiveDashboard",
    "_render_line_plot",
]
