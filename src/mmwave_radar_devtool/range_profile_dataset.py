"""Range-profile preprocessing utilities for TI xWR18xx/xWR68xx DCA1000 captures."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .cfg_parser import RadarCliConfig, parse_radar_cfg
from .exceptions import ConfigurationError

C_M_PER_S = 299_792_458.0


@dataclass(slots=True, frozen=True)
class RadarTensorConfig:
    """Capture layout and profile parameters needed for tensor reconstruction."""

    chirps_per_frame: int
    num_rx: int
    num_tx: int
    num_adc_samples: int
    adc_sample_rate_ksps: float
    freq_slope_mhz_per_us: float
    start_freq_ghz: float
    idle_time_us: float
    ramp_end_time_us: float

    @property
    def complex_samples_per_frame(self) -> int:
        """Return number of complex samples in one frame."""
        return self.chirps_per_frame * self.num_rx * self.num_adc_samples

    @property
    def bytes_per_frame(self) -> int:
        """Return expected frame size in bytes for complex int16 I/Q."""
        return self.complex_samples_per_frame * 4

    @property
    def adc_sample_rate_hz(self) -> float:
        """Return sample rate in Hz."""
        return self.adc_sample_rate_ksps * 1_000.0

    @property
    def freq_slope_hz_per_s(self) -> float:
        """Return FMCW slope in Hz/s."""
        return self.freq_slope_mhz_per_us * 1e12


@dataclass(slots=True, frozen=True)
class ProcessedRangeTensors:
    """Computed tensors derived from the reconstructed radar cube."""

    cube: np.ndarray
    range_cube: np.ndarray
    magnitude: np.ndarray
    power: np.ndarray
    log_magnitude_db: np.ndarray
    log_power_db: np.ndarray
    doppler_cube: np.ndarray
    zero_doppler_power: np.ndarray
    zero_doppler_db: np.ndarray
    range_doppler_power_unshifted: np.ndarray
    range_doppler_power_shifted: np.ndarray


def parse_tensor_cfg(cfg_path: str | Path) -> RadarTensorConfig:
    """Parse TI cfg file into tensor reconstruction parameters."""
    cfg = parse_radar_cfg(cfg_path)
    return _config_from_parsed_cfg(cfg)


def _config_from_parsed_cfg(cfg: RadarCliConfig) -> RadarTensorConfig:
    """Build tensor config from parsed cfg object."""
    profile = cfg.parse_profile_cfg()
    channel = cfg.parse_channel_cfg()

    frame_line = cfg.find_first("frameCfg")
    if frame_line is None:
        raise ConfigurationError("Configuration file does not contain frameCfg.")
    frame_parts = frame_line.text.split()
    if len(frame_parts) < 4:
        raise ConfigurationError(f"Malformed frameCfg command: {frame_line.text}")
    try:
        chirp_start_idx = int(frame_parts[1])
        chirp_end_idx = int(frame_parts[2])
        num_loops = int(frame_parts[3])
    except ValueError as exc:
        raise ConfigurationError(f"Malformed frameCfg numeric fields: {frame_line.text}") from exc

    chirps_per_loop = chirp_end_idx - chirp_start_idx + 1
    chirps_per_frame = chirps_per_loop * num_loops
    if chirps_per_frame <= 0:
        raise ConfigurationError(f"Invalid chirps per frame from frameCfg: {frame_line.text}")

    num_rx = channel.num_enabled_rx
    if num_rx <= 0:
        raise ConfigurationError("channelCfg enables zero RX channels.")
    num_tx = int(bin(channel.tx_channel_enable_mask & 0x7).count("1"))
    if num_tx <= 0:
        num_tx = 1

    return RadarTensorConfig(
        chirps_per_frame=chirps_per_frame,
        num_rx=num_rx,
        num_tx=num_tx,
        num_adc_samples=profile.num_adc_samples,
        adc_sample_rate_ksps=float(profile.dig_out_sample_rate_ksps),
        freq_slope_mhz_per_us=float(profile.freq_slope_mhz_per_us),
        start_freq_ghz=float(profile.start_frequency_ghz),
        idle_time_us=float(profile.idle_time_us),
        ramp_end_time_us=float(profile.ramp_end_time_us),
    )


def load_raw_words(path: str | Path) -> np.ndarray:
    """Load a DCA1000 capture as little-endian signed int16 words."""
    words = np.fromfile(Path(path), dtype="<i2")
    if words.size == 0:
        raise ValueError(f"Input capture is empty: {path}")
    return words


def decode_complex_xwr18xx(words: np.ndarray) -> np.ndarray:
    """Decode complex samples from TI xWR18xx DCA1000 word ordering.

    Ordering per 4-word group:
    I0, I1, Q0, Q1, I2, I3, Q2, Q3, ...
    """
    if words.size % 4 != 0:
        raise ValueError(
            f"Raw word count must be divisible by 4 for I0 I1 Q0 Q1 grouping, got {words.size}."
        )
    groups = words.reshape(-1, 4).astype(np.float32)
    i_values = groups[:, :2].reshape(-1)
    q_values = groups[:, 2:].reshape(-1)
    return (i_values + 1j * q_values).astype(np.complex64)


def reshape_to_cube(
    complex_stream: np.ndarray,
    cfg: RadarTensorConfig,
    *,
    allow_trailing_partial_frame: bool = False,
    capture_label: str | None = None,
    max_trailing_drop_fraction: float = 0.08,
) -> np.ndarray:
    """Reshape complex stream to [frame, chirp, rx, adc_sample]."""
    samples_per_frame = cfg.complex_samples_per_frame
    remainder = complex_stream.size % samples_per_frame
    if remainder != 0:
        if allow_trailing_partial_frame:
            usable_samples = complex_stream.size - remainder
            if usable_samples <= 0:
                raise ValueError(
                    "Complex sample count does not contain one full frame after trimming. "
                    f"got={complex_stream.size}, required-frame-size={samples_per_frame}"
                )
            drop_fraction = float(remainder) / float(max(1, complex_stream.size))
            if drop_fraction > float(max_trailing_drop_fraction):
                raise ValueError(
                    "Capture has large trailing partial frame; refusing to trim. "
                    f"capture={capture_label or '<unknown>'} "
                    f"dropped_complex_samples={remainder} "
                    f"total_complex_samples={complex_stream.size} "
                    f"drop_fraction={drop_fraction * 100.0:.3f}% "
                    f"threshold={float(max_trailing_drop_fraction) * 100.0:.3f}%"
                )
            warnings.warn(
                "Capture has trailing partial frame; "
                f"capture={capture_label or '<unknown>'} "
                f"dropping {remainder} complex samples "
                f"({drop_fraction * 100.0:.4f}% of total).",
                RuntimeWarning,
                stacklevel=2,
            )
            complex_stream = complex_stream[:usable_samples]
        else:
            raise ValueError(
                "Complex sample count is not divisible by chirps*rx*samples-per-chirp. "
                f"got={complex_stream.size}, required-multiple={samples_per_frame}"
            )
    if complex_stream.size == 0:
        raise ValueError(
            "Complex sample count does not contain one full frame. "
            f"got=0, required-frame-size={samples_per_frame}"
        )
    num_frames = complex_stream.size // samples_per_frame
    return complex_stream.reshape(
        num_frames,
        cfg.chirps_per_frame,
        cfg.num_rx,
        cfg.num_adc_samples,
    )


def make_range_window(num_adc_samples: int, kind: str = "hann") -> np.ndarray:
    """Build window vector for range FFT."""
    if kind == "hann":
        return np.hanning(num_adc_samples).astype(np.float32)
    if kind == "rect":
        return np.ones(num_adc_samples, dtype=np.float32)
    raise ValueError(f"Unsupported window kind: {kind}")


def process_capture(
    capture_path: str | Path,
    cfg: RadarTensorConfig,
    *,
    window_kind: str = "hann",
    eps: float = 1e-9,
    allow_trailing_partial_iq: bool = False,
    allow_trailing_partial_frame: bool = False,
    max_trailing_drop_fraction: float = 0.08,
) -> ProcessedRangeTensors:
    """Process one capture into range and range-Doppler tensors."""
    if max_trailing_drop_fraction < 0.0:
        raise ValueError(
            "max_trailing_drop_fraction must be >= 0.0, "
            f"got {max_trailing_drop_fraction}"
        )
    words = load_raw_words(capture_path)
    if allow_trailing_partial_iq and words.size % 4 != 0:
        usable_words = (words.size // 4) * 4
        if usable_words <= 0:
            raise ValueError(
                "Capture does not contain enough complete I/Q words after trimming. "
                f"got={words.size}, capture={capture_path}"
            )
        dropped_words = int(words.size - usable_words)
        drop_fraction = float(dropped_words) / float(max(1, words.size))
        if drop_fraction > float(max_trailing_drop_fraction):
            raise ValueError(
                "Capture has large trailing partial I/Q group; refusing to trim. "
                f"capture={capture_path} dropped_words={dropped_words} "
                f"total_words={words.size} drop_fraction={drop_fraction * 100.0:.3f}% "
                f"threshold={float(max_trailing_drop_fraction) * 100.0:.3f}%"
            )
        warnings.warn(
            "Capture has trailing partial I/Q group; "
            f"capture={capture_path} dropping {dropped_words} int16 words "
            f"({drop_fraction * 100.0:.4f}% of total).",
            RuntimeWarning,
            stacklevel=2,
        )
        words = words[:usable_words]

    complex_stream = decode_complex_xwr18xx(words)
    cube = reshape_to_cube(
        complex_stream,
        cfg,
        allow_trailing_partial_frame=allow_trailing_partial_frame,
        capture_label=str(capture_path),
        max_trailing_drop_fraction=max_trailing_drop_fraction,
    )

    window = make_range_window(cfg.num_adc_samples, kind=window_kind)
    range_cube = np.fft.fft(cube * window[None, None, None, :], axis=-1)
    magnitude = np.abs(range_cube).astype(np.float32)
    power = (magnitude**2).astype(np.float32)
    log_magnitude_db = (20.0 * np.log10(magnitude + eps)).astype(np.float32)
    log_power_db = (10.0 * np.log10(power + eps)).astype(np.float32)

    doppler_cube = np.fft.fft(range_cube, axis=1)
    zero_doppler_power = np.mean(np.abs(doppler_cube[:, 0, :, :]) ** 2, axis=1).astype(np.float32)
    zero_doppler_db = (10.0 * np.log10(zero_doppler_power + eps)).astype(np.float32)

    rd_power_chirp_major = np.mean(np.abs(doppler_cube) ** 2, axis=2).astype(np.float32)
    range_doppler_power_unshifted = np.transpose(rd_power_chirp_major, (0, 2, 1))
    range_doppler_power_shifted = np.fft.fftshift(range_doppler_power_unshifted, axes=2).astype(
        np.float32
    )

    return ProcessedRangeTensors(
        cube=cube,
        range_cube=range_cube,
        magnitude=magnitude,
        power=power,
        log_magnitude_db=log_magnitude_db,
        log_power_db=log_power_db,
        doppler_cube=doppler_cube,
        zero_doppler_power=zero_doppler_power,
        zero_doppler_db=zero_doppler_db,
        range_doppler_power_unshifted=range_doppler_power_unshifted,
        range_doppler_power_shifted=range_doppler_power_shifted,
    )


def compute_range_axis_m(cfg: RadarTensorConfig) -> np.ndarray:
    """Compute range axis in meters for full complex FFT bins."""
    freqs = np.fft.fftfreq(cfg.num_adc_samples, d=1.0 / cfg.adc_sample_rate_hz)
    return (C_M_PER_S * freqs) / (2.0 * cfg.freq_slope_hz_per_s)


def detect_target_bin_candidates(
    range_profile: np.ndarray,
    cfg: RadarTensorConfig,
    *,
    target_m: float = 0.4,
    search_radius: int = 2,
) -> dict[str, float]:
    """Detect target peak around expected bin and mirrored bin."""
    if range_profile.ndim != 1:
        raise ValueError("range_profile must be one-dimensional.")
    n = range_profile.size
    bin_float = (2.0 * cfg.freq_slope_hz_per_s * target_m / C_M_PER_S) * (
        n / cfg.adc_sample_rate_hz
    )
    expected_bin = round(bin_float) % n
    mirror_bin = (n - expected_bin) % n

    def _local_peak(center: int) -> tuple[int, float]:
        lo = max(0, center - search_radius)
        hi = min(n, center + search_radius + 1)
        local = range_profile[lo:hi]
        peak_offset = int(np.argmax(local))
        peak_bin = lo + peak_offset
        return peak_bin, float(range_profile[peak_bin])

    pos_bin, pos_value = _local_peak(expected_bin)
    neg_bin, neg_value = _local_peak(mirror_bin)
    return {
        "expected_positive_bin": float(expected_bin),
        "expected_mirror_bin": float(mirror_bin),
        "detected_positive_bin": float(pos_bin),
        "detected_positive_value": pos_value,
        "detected_mirror_bin": float(neg_bin),
        "detected_mirror_value": neg_value,
    }


def create_training_inputs(
    object_logmag: np.ndarray,
    *,
    empty_logmag: np.ndarray | None = None,
    window_frames: int | None = None,
    window_step: int = 1,
) -> np.ndarray:
    """Create NN-ready tensors from log-magnitude data.

    Returns:
        - [frames, chirps, rx, bins] if window_frames is None
        - [num_windows, window_frames, chirps, rx, bins] otherwise
    """
    x = np.asarray(object_logmag, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected object_logmag shape [F,C,R,B], got {x.shape}.")
    if empty_logmag is not None:
        empty = np.asarray(empty_logmag, dtype=np.float32)
        if empty.ndim != 4:
            raise ValueError(f"Expected empty_logmag shape [F,C,R,B], got {empty.shape}.")
        background = np.mean(empty, axis=0, keepdims=True)
        x = x - background

    if window_frames is None:
        return x
    if window_frames <= 0:
        raise ValueError("window_frames must be > 0.")
    if window_step <= 0:
        raise ValueError("window_step must be > 0.")
    if x.shape[0] < window_frames:
        raise ValueError(
            f"Not enough frames for windowing: frames={x.shape[0]}, window_frames={window_frames}."
        )

    windows = [
        x[start : start + window_frames]
        for start in range(0, x.shape[0] - window_frames + 1, window_step)
    ]
    return np.stack(windows, axis=0).astype(np.float32)


def validate_capture_layout(
    capture_path: str | Path,
    cfg: RadarTensorConfig,
    *,
    print_fn: Callable[[str], None] | None = print,
) -> int:
    """Validate frame divisibility and return inferred frame count."""
    path = Path(capture_path)
    size_bytes = path.stat().st_size
    expected_bytes_per_frame = cfg.bytes_per_frame
    if print_fn is not None:
        print_fn(f"expected_bytes_per_frame={expected_bytes_per_frame}")
        print_fn(f"capture_size_bytes={size_bytes}")
    if size_bytes % expected_bytes_per_frame != 0:
        raise ValueError(
            "Capture size is not divisible by expected bytes/frame. "
            f"size={size_bytes}, bytes_per_frame={expected_bytes_per_frame}"
        )
    num_frames = size_bytes // expected_bytes_per_frame
    if print_fn is not None:
        print_fn(f"inferred_num_frames={num_frames}")
    return int(num_frames)


def select_useful_range_side(values: np.ndarray, *, side: str = "positive") -> np.ndarray:
    """Select one half of full complex-FFT range bins after side inspection."""
    data = np.asarray(values)
    n = data.shape[-1]
    half = n // 2
    if side == "positive":
        return data[..., :half]
    if side == "negative":
        return data[..., half:]
    raise ValueError("side must be 'positive' or 'negative'.")


def plot_range_profile_slice(
    range_cube: np.ndarray,
    *,
    frame_idx: int,
    chirp_idx: int,
    rx_idx: int,
    output_path: str | Path,
    eps: float = 1e-9,
) -> None:
    """Plot full 256-bin range profile for one frame/chirp/rx."""
    spectrum = np.asarray(range_cube[frame_idx, chirp_idx, rx_idx, :], dtype=np.complex64)
    db = 20.0 * np.log10(np.abs(spectrum) + eps)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(db.size), db, linewidth=1.2)
    ax.set_title(f"Range Profile Slice f={frame_idx} c={chirp_idx} rx={rx_idx}")
    ax.set_xlabel("Range FFT Bin")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.25)
    fig.savefig(Path(output_path), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_zero_doppler_average(
    zero_doppler_db: np.ndarray,
    *,
    output_path: str | Path,
    range_axis_m: np.ndarray | None = None,
) -> None:
    """Plot zero-Doppler range profile averaged over frames."""
    profile = np.mean(np.asarray(zero_doppler_db, dtype=np.float32), axis=0)
    x = np.arange(profile.size) if range_axis_m is None else range_axis_m
    xlabel = "Range FFT Bin" if range_axis_m is None else "Range (m)"
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, profile, linewidth=1.3)
    ax.set_title("Zero-Doppler Range Profile (Average Over Frames)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.25)
    fig.savefig(Path(output_path), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_range_doppler_heatmap(
    range_doppler_power: np.ndarray,
    *,
    frame_idx: int,
    output_path: str | Path,
    eps: float = 1e-9,
) -> None:
    """Plot one frame of range-Doppler power as dB heatmap."""
    frame = np.asarray(range_doppler_power[frame_idx], dtype=np.float32)
    db = 10.0 * np.log10(frame + eps)
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(db, aspect="auto", origin="lower")
    ax.set_title(f"Range-Doppler Heatmap (frame={frame_idx})")
    ax.set_xlabel("Doppler Bin")
    ax.set_ylabel("Range Bin")
    fig.colorbar(image, ax=ax, label="Power (dB)")
    fig.savefig(Path(output_path), dpi=160, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "C_M_PER_S",
    "ProcessedRangeTensors",
    "RadarTensorConfig",
    "compute_range_axis_m",
    "create_training_inputs",
    "decode_complex_xwr18xx",
    "detect_target_bin_candidates",
    "load_raw_words",
    "make_range_window",
    "parse_tensor_cfg",
    "plot_range_doppler_heatmap",
    "plot_range_profile_slice",
    "plot_zero_doppler_average",
    "process_capture",
    "reshape_to_cube",
    "select_useful_range_side",
    "validate_capture_layout",
]
