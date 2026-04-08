
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


C_M_PER_S: float = 299_792_458.0


@dataclass(frozen=True)
class RadarConfig:
    """Radar capture parameters required to reshape and interpret raw samples."""

    num_rx: int
    num_tx: int
    samples_per_chirp: int
    chirps_per_frame: int
    frame_period_ms: float | None
    adc_sample_rate_ksps: float | None
    freq_slope_mhz_per_us: float | None
    start_freq_ghz: float | None
    idle_time_us: float | None
    ramp_end_time_us: float | None
    complex_iq: bool = True

    @property
    def adc_sample_rate_hz(self) -> float | None:
        """Return the ADC sample rate in Hz."""
        if self.adc_sample_rate_ksps is None:
            return None
        return self.adc_sample_rate_ksps * 1_000.0

    @property
    def freq_slope_hz_per_s(self) -> float | None:
        """Return the FMCW slope in Hz/s."""
        if self.freq_slope_mhz_per_us is None:
            return None
        return self.freq_slope_mhz_per_us * 1e12


def _enabled_rx_count(mask: int) -> int:
    """Count enabled receiver channels from a bit mask."""
    return int(bin(mask & 0xF).count("1"))


def _enabled_tx_count(mask: int) -> int:
    """Count enabled transmitter channels from a bit mask."""
    return int(bin(mask & 0x7).count("1"))


def parse_cfg(path: Path) -> RadarConfig:
    """Parse a TI mmWave cfg file and extract the key acquisition parameters."""
    rx_mask: int | None = None
    tx_mask: int | None = None
    samples_per_chirp: int | None = None
    adc_sample_rate_ksps: float | None = None
    freq_slope_mhz_per_us: float | None = None
    start_freq_ghz: float | None = None
    idle_time_us: float | None = None
    ramp_end_time_us: float | None = None
    chirps_per_frame: int | None = None
    frame_period_ms: float | None = None

    chirp_tx_masks: list[int] = []

    for raw_line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue

        parts = line.split()
        command = parts[0]

        if command == "channelCfg" and len(parts) >= 3:
            rx_mask = int(parts[1])
            tx_mask = int(parts[2])

        elif command == "profileCfg" and len(parts) >= 12:
            start_freq_ghz = float(parts[2])
            idle_time_us = float(parts[3])
            ramp_end_time_us = float(parts[5])
            freq_slope_mhz_per_us = float(parts[8])
            samples_per_chirp = int(parts[10])
            adc_sample_rate_ksps = float(parts[11])

        elif command == "chirpCfg" and len(parts) >= 9:
            chirp_tx_masks.append(int(parts[8]))

        elif command == "frameCfg" and len(parts) >= 7:
            start_idx = int(parts[1])
            end_idx = int(parts[2])
            loops = int(parts[3])
            frame_period_ms = float(parts[5])
            chirps_per_loop = end_idx - start_idx + 1
            chirps_per_frame = chirps_per_loop * loops

    inferred_tx = max((_enabled_tx_count(mask) for mask in chirp_tx_masks), default=0)
    num_tx = inferred_tx or (_enabled_tx_count(tx_mask) if tx_mask is not None else 3)
    num_rx = _enabled_rx_count(rx_mask) if rx_mask is not None else 4

    if samples_per_chirp is None:
        raise ValueError("Could not infer samples_per_chirp from cfg.")
    if chirps_per_frame is None:
        raise ValueError("Could not infer chirps_per_frame from cfg.")

    return RadarConfig(
        num_rx=num_rx,
        num_tx=max(1, num_tx),
        samples_per_chirp=samples_per_chirp,
        chirps_per_frame=chirps_per_frame,
        frame_period_ms=frame_period_ms,
        adc_sample_rate_ksps=adc_sample_rate_ksps,
        freq_slope_mhz_per_us=freq_slope_mhz_per_us,
        start_freq_ghz=start_freq_ghz,
        idle_time_us=idle_time_us,
        ramp_end_time_us=ramp_end_time_us,
        complex_iq=True,
    )


def build_config_from_args(args: argparse.Namespace) -> RadarConfig:
    """Create a RadarConfig from either cfg parsing or explicit CLI values."""
    if args.cfg is not None:
        parsed = parse_cfg(args.cfg)
        return RadarConfig(
            num_rx=args.num_rx or parsed.num_rx,
            num_tx=args.num_tx or parsed.num_tx,
            samples_per_chirp=args.samples_per_chirp or parsed.samples_per_chirp,
            chirps_per_frame=args.chirps_per_frame or parsed.chirps_per_frame,
            frame_period_ms=args.frame_period_ms if args.frame_period_ms is not None else parsed.frame_period_ms,
            adc_sample_rate_ksps=args.adc_sample_rate_ksps if args.adc_sample_rate_ksps is not None else parsed.adc_sample_rate_ksps,
            freq_slope_mhz_per_us=args.freq_slope_mhz_per_us if args.freq_slope_mhz_per_us is not None else parsed.freq_slope_mhz_per_us,
            start_freq_ghz=args.start_freq_ghz if args.start_freq_ghz is not None else parsed.start_freq_ghz,
            idle_time_us=args.idle_time_us if args.idle_time_us is not None else parsed.idle_time_us,
            ramp_end_time_us=args.ramp_end_time_us if args.ramp_end_time_us is not None else parsed.ramp_end_time_us,
            complex_iq=not args.real_only,
        )

    required = (
        args.num_rx,
        args.samples_per_chirp,
        args.chirps_per_frame,
    )
    if any(value is None for value in required):
        raise ValueError(
            "Without --cfg, the following arguments are required: "
            "--num-rx, --samples-per-chirp, --chirps-per-frame."
        )

    return RadarConfig(
        num_rx=int(args.num_rx),
        num_tx=int(args.num_tx or 1),
        samples_per_chirp=int(args.samples_per_chirp),
        chirps_per_frame=int(args.chirps_per_frame),
        frame_period_ms=args.frame_period_ms,
        adc_sample_rate_ksps=args.adc_sample_rate_ksps,
        freq_slope_mhz_per_us=args.freq_slope_mhz_per_us,
        start_freq_ghz=args.start_freq_ghz,
        idle_time_us=args.idle_time_us,
        ramp_end_time_us=args.ramp_end_time_us,
        complex_iq=not args.real_only,
    )


def load_raw_capture(path: Path, config: RadarConfig) -> np.ndarray:
    """Load a raw DCA1000-like capture and reshape it as [frame, chirp, rx, sample]."""
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size == 0:
        raise ValueError("The input .bin file is empty.")

    if config.complex_iq:
        if raw.size % 2 != 0:
            raw = raw[:-1]
        raw_complex = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    else:
        raw_complex = raw.astype(np.float32).astype(np.complex64)

    samples_per_chirp_all_rx = config.samples_per_chirp * config.num_rx
    chirps_per_frame = config.chirps_per_frame
    frame_size = samples_per_chirp_all_rx * chirps_per_frame

    usable = (raw_complex.size // frame_size) * frame_size
    if usable == 0:
        raise ValueError(
            "The file is too small to form even one full frame with the provided configuration."
        )

    raw_complex = raw_complex[:usable]
    frames = raw_complex.reshape((-1, chirps_per_frame, config.num_rx, config.samples_per_chirp))
    return frames


def _range_bin_count(config: RadarConfig) -> int:
    """Return the number of bins in the range spectrum for the configured sample mode."""
    if config.complex_iq:
        return config.samples_per_chirp
    return (config.samples_per_chirp // 2) + 1


def compute_range_axis_m(frames: np.ndarray, config: RadarConfig) -> np.ndarray | None:
    """Compute the range axis in meters for a range FFT."""
    if config.adc_sample_rate_hz is None or config.freq_slope_hz_per_s is None:
        return None

    num_samples = frames.shape[-1]
    if config.complex_iq:
        freqs = np.fft.fftfreq(num_samples, d=1.0 / config.adc_sample_rate_hz)
        positive = freqs >= 0.0
        freqs = freqs[positive]
    else:
        freqs = np.fft.rfftfreq(num_samples, d=1.0 / config.adc_sample_rate_hz)
    ranges = (C_M_PER_S * freqs) / (2.0 * config.freq_slope_hz_per_s)
    return ranges


def average_raw_by_rx(frames: np.ndarray) -> np.ndarray:
    """Return a representative time-domain waveform per RX averaged over frames and chirps."""
    mean_chirp = frames.mean(axis=(0, 1))
    return mean_chirp


def average_intensity_by_rx(frames: np.ndarray) -> np.ndarray:
    """Return a representative magnitude envelope per RX."""
    return np.abs(frames).mean(axis=(0, 1))


def _range_fft_last_axis(data: np.ndarray, complex_iq: bool) -> np.ndarray:
    """Compute a range FFT across the last axis and keep non-negative bins."""
    if complex_iq:
        spectrum = np.fft.fft(data, axis=-1)
        positive_bins = spectrum[..., : spectrum.shape[-1] // 2]
        return positive_bins
    return np.fft.rfft(data, axis=-1)


def average_spectrum_by_rx(frames: np.ndarray, config: RadarConfig) -> np.ndarray:
    """Return the average magnitude spectrum per RX."""
    window = np.hanning(frames.shape[-1]).astype(np.float32)
    windowed = frames * window[None, None, None, :]
    spectrum = _range_fft_last_axis(windowed, config.complex_iq)
    magnitude = np.abs(spectrum)
    return magnitude.mean(axis=(0, 1))


def average_range_profile_by_rx(frames: np.ndarray, config: RadarConfig) -> np.ndarray:
    """Return the average range FFT magnitude per RX."""
    return average_spectrum_by_rx(frames, config)


def compute_range_doppler(frames: np.ndarray, rx_index: int, config: RadarConfig) -> np.ndarray:
    """Compute a simple range-Doppler magnitude map for one RX channel."""
    rx_cube = frames[:, :, rx_index, :]
    flattened = rx_cube.reshape((-1, rx_cube.shape[-1]))
    window_range = np.hanning(flattened.shape[-1]).astype(np.float32)
    range_fft = _range_fft_last_axis(flattened * window_range[None, :], config.complex_iq)

    window_doppler = np.hanning(range_fft.shape[0]).astype(np.float32)
    doppler_in = range_fft * window_doppler[:, None]
    rd = np.fft.fftshift(np.fft.fft(doppler_in, axis=0), axes=0)
    return np.abs(rd)


def _make_output_dir(path: Path | None) -> Path | None:
    """Create the output directory when needed."""
    if path is None:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_or_show(fig: plt.Figure, output_dir: Path | None, filename: str, show: bool) -> None:
    """Save a figure and optionally display it."""
    if output_dir is not None:
        fig.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def plot_rx_lines(
    x: np.ndarray,
    y_by_rx: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output_dir: Path | None,
    filename: str,
    show: bool,
) -> None:
    """Plot one line per RX channel."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for rx_index in range(y_by_rx.shape[0]):
        ax.plot(x, y_by_rx[rx_index], linewidth=1.25, label=f"RX{rx_index + 1}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_or_show(fig, output_dir, filename, show)


def plot_combined_line(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output_dir: Path | None,
    filename: str,
    show: bool,
) -> None:
    """Plot a single combined line."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    _save_or_show(fig, output_dir, filename, show)


def plot_range_doppler_map(
    rd_map: np.ndarray,
    output_dir: Path | None,
    filename: str,
    show: bool,
) -> None:
    """Plot a range-Doppler magnitude heatmap."""
    fig, ax = plt.subplots(figsize=(12, 7))
    log_rd = 20.0 * np.log10(rd_map + 1e-9)
    image = ax.imshow(log_rd, aspect="auto", origin="lower")
    ax.set_title("Range-Doppler map")
    ax.set_xlabel("Range bin")
    ax.set_ylabel("Doppler bin")
    fig.colorbar(image, ax=ax, label="Magnitude (dB)")
    _save_or_show(fig, output_dir, filename, show)


def describe_config(config: RadarConfig, frames: np.ndarray) -> str:
    """Build a textual summary of the interpreted capture layout."""
    return "\n".join(
        [
            "Capture summary",
            f"  frames: {frames.shape[0]}",
            f"  chirps_per_frame: {frames.shape[1]}",
            f"  num_rx: {frames.shape[2]}",
            f"  samples_per_chirp: {frames.shape[3]}",
            f"  num_tx_cfg: {config.num_tx}",
            f"  adc_sample_rate_ksps: {config.adc_sample_rate_ksps}",
            f"  freq_slope_mhz_per_us: {config.freq_slope_mhz_per_us}",
            f"  frame_period_ms: {config.frame_period_ms}",
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a raw radar .bin capture as waveform, intensity, spectrum, "
            "range profile, and range-Doppler views."
        )
    )
    parser.add_argument("bin_file", type=Path, help="Path to the raw .bin capture file.")
    parser.add_argument("--cfg", type=Path, default=None, help="Optional TI cfg file for automatic parameter extraction.")
    parser.add_argument("--num-rx", type=int, default=None, help="Number of enabled RX channels.")
    parser.add_argument("--num-tx", type=int, default=None, help="Number of TX channels implied by the capture pattern.")
    parser.add_argument("--samples-per-chirp", type=int, default=None, help="ADC samples per chirp.")
    parser.add_argument("--chirps-per-frame", type=int, default=None, help="Number of chirps per frame.")
    parser.add_argument("--frame-period-ms", type=float, default=None, help="Frame period in milliseconds.")
    parser.add_argument("--adc-sample-rate-ksps", type=float, default=None, help="ADC sampling rate in ksps.")
    parser.add_argument("--freq-slope-mhz-per-us", type=float, default=None, help="Chirp frequency slope in MHz/us.")
    parser.add_argument("--start-freq-ghz", type=float, default=None, help="Start frequency in GHz.")
    parser.add_argument("--idle-time-us", type=float, default=None, help="Idle time in microseconds.")
    parser.add_argument("--ramp-end-time-us", type=float, default=None, help="Ramp end time in microseconds.")
    parser.add_argument("--real-only", action="store_true", help="Treat the capture as real samples instead of interleaved I/Q.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory to save plots as PNG files.")
    parser.add_argument("--no-show", action="store_true", help="Do not display figures interactively.")
    parser.add_argument("--rx-for-rd", type=int, default=1, help="Receiver channel to use for the range-Doppler map, 1-based.")
    return parser.parse_args()


def main() -> None:
    """Run the offline radar capture visualizer."""
    args = parse_args()
    config = build_config_from_args(args)
    frames = load_raw_capture(args.bin_file, config)
    output_dir = _make_output_dir(args.output_dir)
    show = not args.no_show

    print(describe_config(config, frames))

    raw_by_rx = average_raw_by_rx(frames)
    intensity_by_rx = average_intensity_by_rx(frames)
    spectrum_by_rx = average_spectrum_by_rx(frames, config)
    range_by_rx = average_range_profile_by_rx(frames, config)

    raw_x = np.arange(config.samples_per_chirp)
    plot_rx_lines(
        x=raw_x,
        y_by_rx=raw_by_rx.real,
        title="Raw waveform by RX channel",
        xlabel="Sample index",
        ylabel="Amplitude (I component)",
        output_dir=output_dir,
        filename="raw_waveform_by_rx.png",
        show=show,
    )

    plot_rx_lines(
        x=raw_x,
        y_by_rx=intensity_by_rx,
        title="Intensity envelope by RX channel",
        xlabel="Sample index",
        ylabel="Magnitude",
        output_dir=output_dir,
        filename="intensity_by_rx.png",
        show=show,
    )

    spectrum_x = np.arange(spectrum_by_rx.shape[-1])
    plot_rx_lines(
        x=spectrum_x,
        y_by_rx=20.0 * np.log10(spectrum_by_rx + 1e-9),
        title="Beat spectrum by RX channel",
        xlabel="Frequency bin",
        ylabel="Magnitude (dB)",
        output_dir=output_dir,
        filename="spectrum_by_rx.png",
        show=show,
    )

    range_axis_m = compute_range_axis_m(frames, config)
    if range_axis_m is None:
        range_x = np.arange(range_by_rx.shape[-1])
        range_xlabel = "Range bin"
    else:
        range_x = range_axis_m
        range_xlabel = "Range (m)"

    plot_rx_lines(
        x=range_x,
        y_by_rx=20.0 * np.log10(range_by_rx + 1e-9),
        title="Range profile by RX channel",
        xlabel=range_xlabel,
        ylabel="Magnitude (dB)",
        output_dir=output_dir,
        filename="range_profile_by_rx.png",
        show=show,
    )

    combined_range = range_by_rx.mean(axis=0)
    plot_combined_line(
        x=range_x,
        y=20.0 * np.log10(combined_range + 1e-9),
        title="Combined range profile",
        xlabel=range_xlabel,
        ylabel="Magnitude (dB)",
        output_dir=output_dir,
        filename="range_profile_combined.png",
        show=show,
    )

    rx_for_rd = max(1, min(args.rx_for_rd, config.num_rx)) - 1
    rd_map = compute_range_doppler(frames, rx_for_rd, config)
    plot_range_doppler_map(
        rd_map=rd_map,
        output_dir=output_dir,
        filename=f"range_doppler_rx{rx_for_rd + 1}.png",
        show=show,
    )

    if show:
        plt.show()


if __name__ == "__main__":
    main()
