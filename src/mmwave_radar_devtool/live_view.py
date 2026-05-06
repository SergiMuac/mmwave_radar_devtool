"""Rich live dashboard for radar capture telemetry and range-profile views."""

from __future__ import annotations

import shutil
import sys
import termios
import threading
import time
import tty
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise

import numpy as np
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .cfg_parser import RadarCliConfig
from .dca1000 import DCA1000DataPacket
from .exceptions import ConfigurationError

C_M_PER_S = 299_792_458.0


class SignalViewMode(StrEnum):
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
    y_min: float | None = None
    y_max: float | None = None


@dataclass(slots=True, frozen=True)
class LivePredictionResult:
    """Small prediction payload rendered by the live dashboard."""

    task: str
    primary: str
    confidence: float | None = None
    detail: str = ""


@dataclass(slots=True, frozen=True)
class LiveSignalProcessingConfig:
    """User-tunable processing options for live spectrum/range views."""

    decode_order: str = "iiqq"
    range_window_kind: str = "hann"
    range_side: str = "positive"
    normalize_spectrum_to_peak: bool = False
    normalize_range_to_peak: bool = False
    spectrum_db_min: float | None = 0.0
    spectrum_db_max: float | None = None
    range_db_min: float | None = None
    range_db_max: float | None = None
    chirp_alignment_offset: int = -1
    baseline_range_db: np.ndarray | None = None
    prediction_callback: Callable[[np.ndarray], LivePredictionResult | None] | None = None
    prediction_interval_s: float = 0.5


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
    malformed_datagrams_detected: int = 0
    recent_packet_rates: deque[float] = field(default_factory=lambda: deque(maxlen=180))
    recent_throughput_rates: deque[float] = field(default_factory=lambda: deque(maxlen=180))
    packets_since_last_rate: int = 0
    bytes_since_last_rate: int = 0
    last_rate_timestamp: float = field(default_factory=time.monotonic)
    current_mode: SignalViewMode = SignalViewMode.RAW
    latest_raw: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    latest_complex: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.complex64))
    complex_history: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.complex64))
    max_complex_history: int = 524_288
    complex_history_start_index: int = 0
    decode_order: str = "iiqq"
    q_first: bool = False

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

    def record_malformed_datagram(self, _: int) -> None:
        """Record a malformed datagram that could not be parsed."""
        self.malformed_datagrams_detected += 1

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
        complex_samples = _decode_dca_complex_words(
            samples,
            decode_order=self.decode_order,
            q_first=self.q_first,
        )
        if complex_samples.size == 0:
            return
        if self.complex_history.size == 0:
            updated = complex_samples
        else:
            updated = np.concatenate((self.complex_history, complex_samples))
        if updated.size > self.max_complex_history:
            removed = updated.size - self.max_complex_history
            updated = updated[removed:]
            self.complex_history_start_index += removed
        self.complex_history = updated.astype(np.complex64)
        self.latest_complex = self.complex_history[-4096:]


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
            with suppress(termios.error):
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._original_termios)
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
            elif key in {"q", "Q", "\x03"}:
                self.stop_callback()


class TerminalLiveDashboard:
    """Rich terminal dashboard for live radar telemetry and simple plots."""

    def __init__(
        self,
        radar_cfg: RadarCliConfig | None = None,
        title: str = "IWR1843 • DCA1000 Live",
        processing_config: LiveSignalProcessingConfig | None = None,
    ) -> None:
        """Initialize the dashboard."""
        self._title = title
        self._metrics = LiveMetrics()
        self._console = Console()
        self._live: Live | None = None
        self._active = False
        self._stop_requested = False
        self._profile_cfg = radar_cfg.parse_profile_cfg() if radar_cfg is not None else None
        self._processing = processing_config or LiveSignalProcessingConfig()
        self._rx_count = 1
        self._chirps_per_frame = 16
        self._auto_alignment_enabled = self._processing.chirp_alignment_offset < 0
        self._resolved_chirp_alignment_offset: int | None = None
        if radar_cfg is not None:
            try:
                self._rx_count = max(1, radar_cfg.parse_channel_cfg().num_enabled_rx)
            except ConfigurationError:
                self._rx_count = 1
            self._chirps_per_frame = _parse_chirps_per_frame(radar_cfg) or 16
            try:
                self._metrics.q_first = radar_cfg.parse_adcbuf_cfg().q_first
            except ConfigurationError:
                self._metrics.q_first = False
        self._metrics.decode_order = self._processing.decode_order
        self._y_limits_by_mode: dict[SignalViewMode, tuple[float, float]] = {}
        self._display_limits_by_mode: dict[SignalViewMode, tuple[float, float]] = {}
        self._latest_prediction: LivePredictionResult | None = None
        self._last_prediction_at = 0.0
        self._prediction_error: str | None = None
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
        prediction = self._update_prediction()
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
        summary.add_row(
            _metric_tile("Malformed", f"{self._metrics.malformed_datagrams_detected:,}", "#f97316"),
            _metric_tile("Mode", self._metrics.current_mode.value.upper(), "#f43f5e"),
            _metric_tile("RX", str(self._rx_count), "#a78bfa"),
            _metric_tile("Chirps/frame", str(self._chirps_per_frame), "#eab308"),
        )
        if self._processing.prediction_callback is not None:
            summary.add_row(
                _metric_tile("NN", self._prediction_primary(prediction), "#22c55e"),
                _metric_tile("NN task", self._prediction_task(prediction), "#38bdf8"),
                _metric_tile("Confidence", self._prediction_confidence(prediction), "#f59e0b"),
                _metric_tile("NN detail", self._prediction_detail(prediction), "#c084fc"),
            )

        controls = Panel(
            _render_mode_tabs(self._metrics.current_mode),
            title="Views",
            subtitle=(
                "Keys: 1 Raw   2 Intensity   3 Spectrum   4 Range FFT   q Quit"
                f"   |   Align: {self._alignment_status_label()}"
            ),
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

    def _update_prediction(self) -> LivePredictionResult | None:
        """Run throttled live ML prediction on the latest complete frame."""
        callback = self._processing.prediction_callback
        if callback is None:
            return None

        now = time.monotonic()
        interval = max(0.05, float(self._processing.prediction_interval_s))
        if self._latest_prediction is not None and now - self._last_prediction_at < interval:
            return self._latest_prediction

        frame_db = self._compute_zero_doppler_frame_by_rx_db()
        if frame_db is None:
            return self._latest_prediction

        try:
            prediction = callback(frame_db)
        except Exception as exc:  # pragma: no cover - visible in live UI.
            self._prediction_error = type(exc).__name__
            return self._latest_prediction

        if prediction is not None:
            self._latest_prediction = prediction
            self._last_prediction_at = now
            self._prediction_error = None
        return self._latest_prediction

    def _build_raw_series(self) -> PlotSeries:
        """Build a raw sample waveform plot."""
        values = self._metrics.latest_raw
        if values.size == 0:
            values = np.zeros(128, dtype=np.float32)
            y_min, y_max = -1.0, 1.0
        else:
            y_min, y_max = self._stabilized_limits(SignalViewMode.RAW, values)
        return PlotSeries(
            values=values.astype(np.float32),
            title="Raw ADC waveform",
            left_label="0",
            right_label=f"{values.size} samples",
            unit="adc",
            accent="#22d3ee",
            y_min=y_min,
            y_max=y_max,
        )

    def _build_intensity_series(self) -> PlotSeries:
        """Build a short-window intensity envelope plot."""
        chirp_cube = self._compute_chirp_time_cube()
        if chirp_cube.size == 0:
            values = np.zeros(128, dtype=np.float32)
        else:
            # Show one RX waveform per chirp instead of concatenated RX streams.
            values = np.abs(chirp_cube[-1, 0, :]).astype(np.float32)
            if values.size >= 8:
                kernel = np.ones(8, dtype=np.float32) / 8.0
                values = np.convolve(values.astype(np.float32), kernel, mode="same")
        y_min, y_max = self._stabilized_limits(SignalViewMode.INTENSITY, values)
        return PlotSeries(
            values=values.astype(np.float32),
            title="Intensity envelope (last chirp, RX0)",
            left_label="near",
            right_label="farther samples",
            unit="|IQ|",
            accent="#34d399",
            y_min=y_min,
            y_max=y_max,
        )

    def _build_spectrum_series(self) -> PlotSeries:
        """Build FFT magnitude of the real-valued chirp response, averaged over chirps/RX."""
        spectrum = self._compute_chirp_aligned_spectrum()
        spectrum_db = _magnitude_to_db(
            spectrum,
            normalize_to_peak=self._processing.normalize_spectrum_to_peak,
            floor_db=-120.0,
        )
        y_min, y_max = self._resolve_db_limits(
            SignalViewMode.SPECTRUM,
            spectrum_db,
            configured_min=self._processing.spectrum_db_min,
            configured_max=self._processing.spectrum_db_max,
        )
        return PlotSeries(
            values=spectrum_db.astype(np.float32),
            title="FFT(real response) magnitude (avg chirps/RX)",
            left_label="0 m",
            right_label=self._format_max_frequency_or_range(len(spectrum)),
            unit="dB",
            accent="#c084fc",
            y_min=y_min,
            y_max=y_max,
        )

    def _build_range_series(self) -> PlotSeries:
        """Build a zero-Doppler TI-demo-like range profile."""
        profile = self._compute_zero_doppler_profile()
        baseline = self._processing.baseline_range_db
        if baseline is None:
            spectrum_db = _power_to_db(
                profile,
                normalize_to_peak=self._processing.normalize_range_to_peak,
                floor_db=-120.0,
            )
            title = "Zero-Doppler range profile (RX-mean power)"
        else:
            current_db = _power_to_db(
                profile,
                normalize_to_peak=False,
                floor_db=-120.0,
            )
            baseline_db = np.asarray(baseline, dtype=np.float32)
            use_bins = min(current_db.size, baseline_db.size)
            if use_bins <= 0:
                spectrum_db = np.zeros(128, dtype=np.float32)
            else:
                spectrum_db = np.abs(current_db[:use_bins] - baseline_db[:use_bins]).astype(
                    np.float32
                )
            if self._processing.normalize_range_to_peak and spectrum_db.size > 0:
                spectrum_db = spectrum_db - float(np.max(spectrum_db))
            title = "Zero-Doppler absolute range delta vs baseline (RX-mean power)"
        configured_min = self._processing.range_db_min
        if baseline is not None and configured_min is None:
            configured_min = 0.0
        y_min, y_max = self._resolve_db_limits(
            SignalViewMode.RANGE,
            spectrum_db,
            configured_min=configured_min,
            configured_max=self._processing.range_db_max,
        )
        return PlotSeries(
            values=spectrum_db.astype(np.float32),
            title=title,
            left_label="0 m",
            right_label=self._format_max_range(len(spectrum_db)),
            unit="dB",
            accent="#f59e0b",
            y_min=y_min,
            y_max=y_max,
        )

    def _compute_chirp_aligned_spectrum(self) -> np.ndarray:
        """Compute averaged FFT magnitude using only the real component of chirp signals."""
        chirp_cube = self._compute_chirp_time_cube()
        if chirp_cube.size == 0:
            return np.zeros(128, dtype=np.float32)

        samples_per_chirp = chirp_cube.shape[-1]
        if self._processing.range_window_kind == "rect":
            window = np.ones(samples_per_chirp, dtype=np.float32)
        else:
            window = np.hanning(samples_per_chirp).astype(np.float32)

        real_response = np.real(chirp_cube).astype(np.float32)
        fft_values = np.fft.fft(real_response * window[None, None, :], axis=-1)
        if self._processing.range_side != "full":
            fft_values = _select_range_side(fft_values, side=self._processing.range_side)
        magnitude = np.abs(fft_values)
        spectrum = magnitude.mean(axis=(0, 1))
        return spectrum.astype(np.float32)

    def _compute_zero_doppler_profile(self) -> np.ndarray:
        """Compute TI-demo-like zero-Doppler power profile over range bins."""
        range_cube = self._compute_range_cube()
        if range_cube.size == 0:
            return np.zeros(128, dtype=np.float32)
        if range_cube.shape[0] >= 2:
            doppler_cube = np.fft.fft(range_cube, axis=0)
            zero_doppler_power = np.mean(np.abs(doppler_cube[0, :, :]) ** 2, axis=0)
        else:
            zero_doppler_power = np.mean(np.abs(range_cube[0, :, :]) ** 2, axis=0)
        side = self._processing.range_side
        if side != "full":
            zero_doppler_power = _select_range_side(zero_doppler_power, side=side)
        return zero_doppler_power.astype(np.float32)

    def _compute_zero_doppler_frame_by_rx_db(self) -> np.ndarray | None:
        """Compute latest ML frame as raw zero-Doppler dB with shape `[RX, B]`."""
        range_cube = self._compute_range_cube()
        if range_cube.size == 0 or range_cube.shape[0] < self._chirps_per_frame:
            return None
        doppler_cube = np.fft.fft(range_cube, axis=0)
        power_by_rx = (np.abs(doppler_cube[0, :, :]) ** 2).astype(np.float32)
        return (10.0 * np.log10(power_by_rx + 1e-9)).astype(np.float32)

    def _compute_range_cube(self) -> np.ndarray:
        """Build [chirp, rx, range_bin] complex range cube from latest history."""
        chirp_cube = self._compute_chirp_time_cube()
        if chirp_cube.size == 0:
            return np.zeros((0, self._rx_count, 0), dtype=np.complex64)
        samples_per_chirp = chirp_cube.shape[-1]

        if self._processing.range_window_kind == "rect":
            window = np.ones(samples_per_chirp, dtype=np.float32)
        else:
            window = np.hanning(samples_per_chirp).astype(np.float32)

        range_cube = np.fft.fft(chirp_cube * window[None, None, :], axis=-1)
        side = self._processing.range_side
        if side != "full":
            range_cube = _select_range_side(range_cube, side=side)
        return range_cube.astype(np.complex64)

    def _compute_chirp_time_cube(self) -> np.ndarray:
        """Build [chirp, rx, adc_sample] complex time cube with stable chirp alignment."""
        values = self._metrics.complex_history
        profile = self._profile_cfg
        if values.size == 0:
            return np.zeros((0, self._rx_count, 0), dtype=np.complex64)
        if profile is None:
            return values.reshape(1, 1, -1).astype(np.complex64)

        samples_per_chirp = int(profile.num_adc_samples)
        if samples_per_chirp <= 0:
            return np.zeros((0, self._rx_count, 0), dtype=np.complex64)
        samples_per_chirp_all_rx = samples_per_chirp * self._rx_count
        if samples_per_chirp_all_rx <= 0:
            return np.zeros((0, self._rx_count, 0), dtype=np.complex64)

        history_start = self._metrics.complex_history_start_index
        alignment = self._resolve_chirp_alignment_offset(
            values=values,
            history_start=history_start,
            samples_per_chirp_all_rx=samples_per_chirp_all_rx,
            samples_per_chirp=samples_per_chirp,
        )
        first_local = (
            alignment - (history_start % samples_per_chirp_all_rx)
        ) % samples_per_chirp_all_rx
        if first_local >= values.size:
            return np.zeros((0, self._rx_count, 0), dtype=np.complex64)

        aligned = values[first_local:]
        available_chirps = aligned.size // samples_per_chirp_all_rx
        if available_chirps <= 0:
            return np.zeros((0, self._rx_count, 0), dtype=np.complex64)

        chirps_to_use = min(max(self._chirps_per_frame, 1), available_chirps)
        usable = chirps_to_use * samples_per_chirp_all_rx
        return aligned[-usable:].reshape(chirps_to_use, self._rx_count, samples_per_chirp).astype(
            np.complex64
        )

    def _resolve_chirp_alignment_offset(
        self,
        *,
        values: np.ndarray,
        history_start: int,
        samples_per_chirp_all_rx: int,
        samples_per_chirp: int,
    ) -> int:
        """Resolve chirp alignment offset, auto-estimating when configured."""
        if not self._auto_alignment_enabled:
            return int(self._processing.chirp_alignment_offset) % samples_per_chirp_all_rx

        if self._resolved_chirp_alignment_offset is not None:
            return self._resolved_chirp_alignment_offset

        minimum_chirps_for_estimate = max(self._chirps_per_frame * 8, 128)
        if values.size < minimum_chirps_for_estimate * samples_per_chirp_all_rx:
            return 0

        estimate = self._estimate_chirp_alignment_offset(
            values=values,
            history_start=history_start,
            samples_per_chirp_all_rx=samples_per_chirp_all_rx,
            samples_per_chirp=samples_per_chirp,
        )
        self._resolved_chirp_alignment_offset = estimate
        return estimate

    def _estimate_chirp_alignment_offset(
        self,
        *,
        values: np.ndarray,
        history_start: int,
        samples_per_chirp_all_rx: int,
        samples_per_chirp: int,
    ) -> int:
        """Estimate chirp alignment by minimizing mode-3/4 frame-to-frame instability."""
        step = 16
        window = np.hanning(samples_per_chirp).astype(np.float32)
        eps = 1e-9
        best_score: float | None = None
        best_offset = 0

        for offset in range(0, samples_per_chirp_all_rx, step):
            first_local = (
                offset - (history_start % samples_per_chirp_all_rx)
            ) % samples_per_chirp_all_rx
            if first_local >= values.size:
                continue
            aligned = values[first_local:]
            available_chirps = aligned.size // samples_per_chirp_all_rx
            if available_chirps < self._chirps_per_frame * 2:
                continue
            use_chirps = min(available_chirps, self._chirps_per_frame * 32)
            frame_count = use_chirps // self._chirps_per_frame
            if frame_count < 2:
                continue
            use_chirps = frame_count * self._chirps_per_frame
            tail = aligned[-use_chirps * samples_per_chirp_all_rx :]
            frame_cube = tail.reshape(
                frame_count,
                self._chirps_per_frame,
                self._rx_count,
                samples_per_chirp,
            )
            range_cube = np.fft.fft(frame_cube * window[None, None, None, :], axis=-1)
            if self._processing.range_side != "full":
                range_cube = _select_range_side(range_cube, side=self._processing.range_side)

            # Mode 3 proxy: frame-wise average range FFT magnitude.
            mode3_db = 20.0 * np.log10(np.abs(range_cube).mean(axis=(1, 2)) + eps)

            # Mode 4 proxy: frame-wise zero-Doppler RX-mean power profile.
            doppler_cube = np.fft.fft(range_cube, axis=1)
            zero_doppler_power = np.mean(np.abs(doppler_cube[:, 0, :, :]) ** 2, axis=1)
            mode4_db = 10.0 * np.log10(zero_doppler_power + eps)

            if mode3_db.shape[1] < 8 or mode4_db.shape[1] < 8:
                continue

            flicker3 = float(np.mean(np.abs(np.diff(mode3_db, axis=0))))
            flicker4 = float(np.mean(np.abs(np.diff(mode4_db, axis=0))))
            rough3 = float(np.mean(np.abs(np.diff(mode3_db, n=2, axis=1))))
            rough4 = float(np.mean(np.abs(np.diff(mode4_db, n=2, axis=1))))
            score = flicker4 + 0.6 * flicker3 + 0.2 * rough3 + 0.2 * rough4
            if best_score is None or score < best_score:
                best_score = score
                best_offset = offset

        return best_offset

    def _resolve_db_limits(
        self,
        mode: SignalViewMode,
        values: np.ndarray,
        *,
        configured_min: float | None,
        configured_max: float | None,
    ) -> tuple[float, float]:
        """Resolve dB limits with headroom so peaks do not clip at the top."""
        if configured_min is not None and configured_max is not None:
            fixed_min = float(configured_min)
            fixed_max = float(configured_max)
            if fixed_max <= fixed_min + 1e-3:
                fixed_max = fixed_min + 1.0
            resolved = (fixed_min, fixed_max)
            self._display_limits_by_mode[mode] = resolved
            return resolved

        source = np.asarray(values, dtype=np.float32)
        auto_min, auto_max = self._stabilized_limits(mode, source)
        target_min = auto_min if configured_min is None else float(configured_min)
        target_max = auto_max if configured_max is None else float(configured_max)

        if source.size > 0:
            source_peak = float(np.percentile(source, 99.8))
            required_top = source_peak + 2.5
            if required_top > target_max:
                target_max = required_top

        if target_max <= target_min + 1e-3:
            target_max = target_min + 1.0

        previous = self._display_limits_by_mode.get(mode)
        if previous is None:
            resolved = (target_min, target_max)
        else:
            min_alpha = 0.20
            max_alpha_up = 0.55
            max_alpha_down = 0.08
            new_min = previous[0] + min_alpha * (target_min - previous[0])
            if target_max > previous[1]:
                new_max = previous[1] + max_alpha_up * (target_max - previous[1])
            else:
                new_max = previous[1] + max_alpha_down * (target_max - previous[1])
            if source.size > 0:
                source_peak = float(np.percentile(source, 99.8))
                required_top = source_peak + 2.5
                if required_top > new_max:
                    new_max = required_top
            resolved = (new_min, new_max)

        self._display_limits_by_mode[mode] = resolved
        return resolved

    def _stabilized_limits(
        self, mode: SignalViewMode, values: np.ndarray
    ) -> tuple[float, float]:
        """Compute slowly adapting y-axis limits to avoid jittery autoscaling."""
        source = np.asarray(values, dtype=np.float32)
        if source.size == 0:
            return 0.0, 1.0
        current_min = float(np.percentile(source, 1.0))
        current_max = float(np.percentile(source, 99.0))
        if current_max - current_min < 1e-6:
            center = float(np.mean(source))
            current_min = center - 0.5
            current_max = center + 0.5

        previous = self._y_limits_by_mode.get(mode)
        if previous is None:
            limits = (current_min, current_max)
        else:
            alpha = 0.18
            limits = (
                previous[0] + alpha * (current_min - previous[0]),
                previous[1] + alpha * (current_max - previous[1]),
            )

        self._y_limits_by_mode[mode] = limits
        return limits

    def _format_max_frequency_or_range(self, bin_count: int) -> str:
        """Format the right-edge label for mode 3."""
        profile = self._profile_cfg
        if profile is None:
            return "Nyquist"
        return self._format_max_range(bin_count)

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

    def _alignment_status_label(self) -> str:
        """Return compact alignment status for the dashboard subtitle."""
        if not self._auto_alignment_enabled:
            return str(int(self._processing.chirp_alignment_offset))
        if self._resolved_chirp_alignment_offset is None:
            return "auto(pending)"
        return f"auto({self._resolved_chirp_alignment_offset})"

    def _prediction_primary(self, prediction: LivePredictionResult | None) -> str:
        """Return compact primary prediction text."""
        if prediction is not None:
            return prediction.primary
        if self._prediction_error is not None:
            return self._prediction_error
        return "warming"

    def _prediction_task(self, prediction: LivePredictionResult | None) -> str:
        """Return compact prediction task text."""
        if prediction is None:
            return "-"
        return prediction.task

    def _prediction_confidence(self, prediction: LivePredictionResult | None) -> str:
        """Return compact prediction confidence text."""
        if prediction is None or prediction.confidence is None:
            return "-"
        return f"{prediction.confidence:.3f}"

    def _prediction_detail(self, prediction: LivePredictionResult | None) -> str:
        """Return compact prediction detail text."""
        if prediction is None:
            return "need frame"
        return prediction.detail or "-"


def _decode_dca_complex_words(
    words: np.ndarray,
    *,
    decode_order: str = "iiqq",
    q_first: bool = False,
) -> np.ndarray:
    """Decode complex int16 words from either IIQQ or IQIQ ordering."""
    raw = np.asarray(words, dtype=np.float32)
    if decode_order == "iiqq":
        if raw.size < 4:
            return np.zeros(0, dtype=np.complex64)
        usable = (raw.size // 4) * 4
        groups = raw[:usable].reshape(-1, 4)
        i_values = groups[:, :2].reshape(-1)
        q_values = groups[:, 2:].reshape(-1)
        return (i_values + 1j * q_values).astype(np.complex64)

    if decode_order == "iqiq":
        if raw.size < 2:
            return np.zeros(0, dtype=np.complex64)
        usable = (raw.size // 2) * 2
        pairs = raw[:usable].reshape(-1, 2)
        if q_first:
            q_values = pairs[:, 0]
            i_values = pairs[:, 1]
        else:
            i_values = pairs[:, 0]
            q_values = pairs[:, 1]
        return (i_values + 1j * q_values).astype(np.complex64)

    raise ValueError(f"Unsupported decode_order: {decode_order}")


def _magnitude_to_db(
    values: np.ndarray, *, normalize_to_peak: bool = False, floor_db: float = -120.0
) -> np.ndarray:
    """Convert linear magnitude to dB, optionally normalized to the current peak."""
    magnitude = np.asarray(values, dtype=np.float32)
    db = 20.0 * np.log10(np.maximum(magnitude, 1e-9))
    if normalize_to_peak:
        peak = float(np.max(magnitude)) if magnitude.size else 0.0
        if peak > 1e-9:
            db -= 20.0 * np.log10(peak)
    db = np.maximum(db, floor_db)
    return db.astype(np.float32)


def _power_to_db(
    values: np.ndarray, *, normalize_to_peak: bool = False, floor_db: float = -120.0
) -> np.ndarray:
    """Convert linear power to dB, optionally normalized to the current peak."""
    power = np.asarray(values, dtype=np.float32)
    db = 10.0 * np.log10(np.maximum(power, 1e-9))
    if normalize_to_peak:
        peak = float(np.max(power)) if power.size else 0.0
        if peak > 1e-9:
            db -= 10.0 * np.log10(peak)
    db = np.maximum(db, floor_db)
    return db.astype(np.float32)


def _select_range_side(values: np.ndarray, *, side: str) -> np.ndarray:
    """Select positive or negative range side from full complex-FFT bins."""
    data = np.asarray(values)
    if side == "full":
        return data
    count = data.shape[-1]
    half = max(1, count // 2)
    if side == "positive":
        return data[..., :half]
    if side == "negative":
        return data[..., half:]
    raise ValueError(f"Unsupported range side: {side}")


def _parse_chirps_per_frame(cfg: RadarCliConfig) -> int | None:
    """Parse chirps per frame from frameCfg, if present."""
    line = cfg.find_first("frameCfg")
    if line is None:
        return None
    parts = line.text.split()
    if len(parts) < 4:
        return None
    try:
        chirp_start_idx = int(parts[1])
        chirp_end_idx = int(parts[2])
        num_loops = int(parts[3])
    except ValueError:
        return None
    chirps_per_loop = chirp_end_idx - chirp_start_idx + 1
    chirps_per_frame = chirps_per_loop * num_loops
    if chirps_per_frame <= 0:
        return None
    return chirps_per_frame


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
    if series.y_min is not None and series.y_max is not None:
        minimum = float(series.y_min)
        maximum = float(series.y_max)
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
            max(0, round((1.0 - float(fraction)) * (subpixel_height - 1))),
        )
        points.append((x, y))

    for start, end in pairwise(points):
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
    "LivePredictionResult",
    "LiveSignalProcessingConfig",
    "PlotSeries",
    "SignalViewMode",
    "TerminalLiveDashboard",
    "_render_line_plot",
]
