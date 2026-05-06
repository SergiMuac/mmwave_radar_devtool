"""Shared data utilities for radar ML workflows."""

from __future__ import annotations

import json
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..range_profile_dataset import (
    C_M_PER_S,
    ProcessedRangeTensors,
    RadarTensorConfig,
    compute_range_axis_m,
    make_range_window,
    process_capture,
    select_useful_range_side,
)

DEFAULT_FLAT_LABEL_REGEX = r"^(?P<label>.+)_(?P<index>\d+)$"
DEFAULT_NUMERIC_TARGET_REGEX = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
FEATURE_MODE_ZERO_DOPPLER_DB = "zero_doppler_db"
FEATURE_MODE_COMPLEX_COHERENT = "complex_coherent"
BACKGROUND_SUBTRACTION_NONE = "none"
BACKGROUND_SUBTRACTION_COMPLEX_RANGE = "complex_range"
BACKGROUND_SUBTRACTION_RAW = "raw"
NORMALIZATION_NONE = "none"
NORMALIZATION_PER_RECORDING = "per_recording_standardize"
NORMALIZATION_TRAINSET = "trainset_standardize"

_WARNED_MISSING_BACKGROUND = False


@dataclass(slots=True, frozen=True)
class LabeledCapture:
    """One capture path paired with a categorical label."""

    path: Path
    label: str


@dataclass(slots=True, frozen=True)
class SplitCaptures:
    """Train/val/test splits at capture granularity."""

    train: tuple[LabeledCapture, ...]
    val: tuple[LabeledCapture, ...]
    test: tuple[LabeledCapture, ...]


@dataclass(slots=True, frozen=True)
class BackgroundReference:
    """Optional empty-scene reference tensors for background subtraction."""

    raw_mean_cube: np.ndarray | None
    complex_range_mean: np.ndarray | None


def parse_hidden_dims(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated hidden-dimension list."""
    parts = [token.strip() for token in str(raw).split(",")]
    dims: list[int] = []
    for part in parts:
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"Hidden dimensions must be > 0, got {value}.")
        dims.append(value)
    if not dims:
        raise ValueError("At least one hidden dimension is required.")
    return tuple(dims)


def discover_bin_files(dataset_dir: Path) -> list[Path]:
    """Return all `.bin` files recursively under a dataset directory."""
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return sorted(path for path in root.rglob("*.bin") if path.is_file())


def _compile_pattern(raw: str, what: str) -> re.Pattern[str]:
    try:
        return re.compile(raw)
    except re.error as exc:
        raise ValueError(f"Invalid {what} regex: {raw!r}") from exc


def infer_capture_label(
    capture_path: Path,
    dataset_dir: Path,
    *,
    flat_label_regex: str = DEFAULT_FLAT_LABEL_REGEX,
) -> str | None:
    """Infer capture label from either parent folder or flat filename."""
    root = Path(dataset_dir)
    rel = capture_path.resolve().relative_to(root.resolve())

    if len(rel.parts) >= 2:
        # Use top-level folder as class label for nested datasets.
        return rel.parts[0]

    pattern = _compile_pattern(flat_label_regex, "flat-label")
    match = pattern.match(capture_path.stem)
    if match is None:
        return None
    groups = match.groupdict()
    label_group = groups.get("label")
    if label_group:
        return str(label_group)
    if match.lastindex is not None and match.lastindex >= 1:
        return str(match.group(1))
    return str(match.group(0))


def discover_labeled_captures(
    dataset_dir: Path,
    *,
    flat_label_regex: str = DEFAULT_FLAT_LABEL_REGEX,
) -> list[LabeledCapture]:
    """Discover labeled captures from nested folders or `label_#.bin` files."""
    captures = discover_bin_files(dataset_dir)
    out: list[LabeledCapture] = []
    unknown: list[Path] = []

    for path in captures:
        label = infer_capture_label(path, dataset_dir, flat_label_regex=flat_label_regex)
        if label is None:
            unknown.append(path)
            continue
        out.append(LabeledCapture(path=path, label=label))

    if not out:
        raise RuntimeError(
            "No labeled .bin captures were discovered. Use either nested folders "
            "(dataset/<label>/*.bin) or flat files like <label>_<index>.bin."
        )

    if unknown:
        preview = ", ".join(path.name for path in unknown[:5])
        suffix = "" if len(unknown) <= 5 else f" (+{len(unknown) - 5} more)"
        raise RuntimeError(
            "Could not infer labels for some captures. "
            f"Examples: {preview}{suffix}. "
            f"Adjust --flat-label-regex (current: {flat_label_regex!r})."
        )

    return out


def _split_counts(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0

    n_val = round(n * val_ratio)
    n_test = round(n * test_ratio)

    if n >= 3:
        if val_ratio > 0.0:
            n_val = max(1, n_val)
        if test_ratio > 0.0:
            n_test = max(1, n_test)

    while n_val + n_test > n - 1:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break

    n_train = n - n_val - n_test
    if n_train <= 0:
        n_train = 1
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1

    return n_train, n_val, n_test


def split_captures_stratified(
    captures: list[LabeledCapture],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitCaptures:
    """Split captures by label so each split has balanced classes."""
    grouped: dict[str, list[LabeledCapture]] = {}
    for capture in captures:
        grouped.setdefault(capture.label, []).append(capture)

    rng = random.Random(seed)
    train: list[LabeledCapture] = []
    val: list[LabeledCapture] = []
    test: list[LabeledCapture] = []

    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        n_train, n_val, n_test = _split_counts(len(items), val_ratio, test_ratio)

        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return SplitCaptures(train=tuple(train), val=tuple(val), test=tuple(test))


def _warn_missing_background_once(background_subtraction: str) -> None:
    """Warn only once when background subtraction is requested without empty capture."""
    global _WARNED_MISSING_BACKGROUND
    if _WARNED_MISSING_BACKGROUND:
        return
    _WARNED_MISSING_BACKGROUND = True
    warnings.warn(
        "Background subtraction requested but no --background-capture was provided. "
        f"Proceeding without subtraction (mode={background_subtraction!r}).",
        RuntimeWarning,
        stacklevel=2,
    )


def _process_capture_lenient(
    capture_path: Path,
    cfg: RadarTensorConfig,
    *,
    window_kind: str,
    eps: float,
) -> ProcessedRangeTensors:
    """Process capture while tolerating common trailing partial-frame tails."""
    return process_capture(
        capture_path,
        cfg,
        window_kind=window_kind,
        eps=eps,
        allow_trailing_partial_iq=True,
        allow_trailing_partial_frame=True,
    )


def _resolve_range_side_bins(
    data: np.ndarray,
    *,
    range_side: str,
) -> np.ndarray:
    """Apply range-side selection only when side is not full."""
    if range_side == "full":
        return np.asarray(data)
    return select_useful_range_side(data, side=range_side)


def compute_target_range_gate_slice(
    cfg: RadarTensorConfig,
    *,
    target_range_m: float,
    range_gate_bins: int,
    range_side: str,
) -> tuple[int, int, int, float]:
    """Compute inclusive target bin and [start,end) slice for distance gate."""
    if target_range_m < 0.0:
        raise ValueError(f"target_range_m must be >= 0, got {target_range_m}")
    if range_gate_bins < 0:
        raise ValueError(f"range_gate_bins must be >= 0, got {range_gate_bins}")

    sampled_bandwidth_hz = (
        cfg.freq_slope_hz_per_s * float(cfg.num_adc_samples) / float(cfg.adc_sample_rate_hz)
    )
    if sampled_bandwidth_hz <= 0.0:
        raise ValueError(
            "Invalid sampled bandwidth from config. "
            f"slope={cfg.freq_slope_hz_per_s} adc_samples={cfg.num_adc_samples} "
            f"sample_rate_hz={cfg.adc_sample_rate_hz}"
        )
    range_resolution_m = C_M_PER_S / (2.0 * sampled_bandwidth_hz)
    target_bin_full = round(float(target_range_m) / float(range_resolution_m))

    bins_by_side = _resolve_range_side_bins(
        np.arange(cfg.num_adc_samples, dtype=np.int32),
        range_side=range_side,
    )
    num_bins = int(bins_by_side.size)
    if num_bins <= 0:
        raise RuntimeError(f"No range bins available for range_side={range_side!r}")

    target_bin = int(np.clip(target_bin_full, 0, num_bins - 1))
    start = max(0, target_bin - int(range_gate_bins))
    end = min(num_bins, target_bin + int(range_gate_bins) + 1)
    return int(start), int(end), int(target_bin), float(range_resolution_m)


def load_background_reference(
    background_capture: Path | None,
    cfg: RadarTensorConfig,
    *,
    window_kind: str,
    eps: float,
    background_subtraction: str,
) -> BackgroundReference | None:
    """Load optional empty-scene reference for complex-domain subtraction."""
    if background_capture is None or background_subtraction == BACKGROUND_SUBTRACTION_NONE:
        if background_subtraction != BACKGROUND_SUBTRACTION_NONE and background_capture is None:
            _warn_missing_background_once(background_subtraction)
        return None

    if background_subtraction not in {
        BACKGROUND_SUBTRACTION_NONE,
        BACKGROUND_SUBTRACTION_COMPLEX_RANGE,
        BACKGROUND_SUBTRACTION_RAW,
    }:
        raise ValueError(
            "background_subtraction must be one of "
            "{'none','complex_range','raw'}, "
            f"got {background_subtraction!r}"
        )

    processed = _process_capture_lenient(
        background_capture,
        cfg,
        window_kind=window_kind,
        eps=eps,
    )

    if background_subtraction == BACKGROUND_SUBTRACTION_RAW:
        raw_mean = np.mean(processed.cube, axis=0).astype(np.complex64, copy=False)
        return BackgroundReference(raw_mean_cube=raw_mean, complex_range_mean=None)

    range_mean = np.mean(processed.range_cube, axis=0).astype(np.complex64, copy=False)
    return BackgroundReference(raw_mean_cube=None, complex_range_mean=range_mean)


def compute_chirp_coherence(
    complex_range_cube: np.ndarray,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """Compute chirp coherence: |mean_chirp(X)| / (mean_chirp(|X|)+eps)."""
    x = np.asarray(complex_range_cube, dtype=np.complex64)
    if x.ndim != 4:
        raise ValueError(f"Expected [F,C,RX,B] complex range cube, got {x.shape}")
    numerator = np.abs(np.mean(x, axis=1))
    denominator = np.mean(np.abs(x), axis=1) + float(eps)
    return (numerator / denominator).astype(np.float32, copy=False)


def _extract_zero_doppler_by_rx_db_from_processed(
    processed: ProcessedRangeTensors,
    *,
    range_side: str,
    eps: float,
    max_frames: int | None,
) -> np.ndarray:
    """Return per-frame zero-Doppler dB from preprocessed tensors."""
    frame_power = np.abs(processed.doppler_cube[:, 0, :, :]) ** 2
    frame_power = frame_power.astype(np.float32)
    frame_power = _resolve_range_side_bins(frame_power, range_side=range_side)
    if max_frames is not None:
        frame_power = frame_power[: max(0, int(max_frames))]
    return (10.0 * np.log10(frame_power + float(eps))).astype(np.float32)


def extract_frame_complex_coherent_by_rx_reim(
    capture_path: Path,
    cfg: RadarTensorConfig,
    *,
    range_side: str,
    window_kind: str,
    eps: float,
    max_frames: int | None,
    target_range_m: float,
    range_gate_bins: int,
    background_subtraction: str = BACKGROUND_SUBTRACTION_NONE,
    background_reference: BackgroundReference | None = None,
) -> np.ndarray:
    """Return coherent complex range features as `[F,RX,B,2]` (Re, Im)."""
    processed = _process_capture_lenient(
        capture_path,
        cfg,
        window_kind=window_kind,
        eps=eps,
    )
    range_cube = np.asarray(processed.range_cube, dtype=np.complex64)

    if background_subtraction == BACKGROUND_SUBTRACTION_RAW:
        if background_reference is not None and background_reference.raw_mean_cube is not None:
            raw_mean = np.asarray(background_reference.raw_mean_cube, dtype=np.complex64)
            if raw_mean.shape != processed.cube.shape[1:]:
                raise ValueError(
                    "Raw background mean shape mismatch for subtraction. "
                    f"background={raw_mean.shape} expected={processed.cube.shape[1:]}"
                )
            adjusted_cube = processed.cube - raw_mean[None, ...]
            window = make_range_window(cfg.num_adc_samples, kind=window_kind)
            range_cube = np.fft.fft(adjusted_cube * window[None, None, None, :], axis=-1).astype(
                np.complex64
            )
        else:
            _warn_missing_background_once(BACKGROUND_SUBTRACTION_RAW)
    elif background_subtraction == BACKGROUND_SUBTRACTION_COMPLEX_RANGE:
        if background_reference is not None and background_reference.complex_range_mean is not None:
            range_mean = np.asarray(background_reference.complex_range_mean, dtype=np.complex64)
            if range_mean.shape != range_cube.shape[1:]:
                raise ValueError(
                    "Complex-range background mean shape mismatch for subtraction. "
                    f"background={range_mean.shape} expected={range_cube.shape[1:]}"
                )
            range_cube = range_cube - range_mean[None, ...]
        else:
            _warn_missing_background_once(BACKGROUND_SUBTRACTION_COMPLEX_RANGE)
    elif background_subtraction != BACKGROUND_SUBTRACTION_NONE:
        raise ValueError(
            "background_subtraction must be one of "
            "{'none','complex_range','raw'}, "
            f"got {background_subtraction!r}"
        )

    range_cube = _resolve_range_side_bins(range_cube, range_side=range_side)
    start, end, _, _ = compute_target_range_gate_slice(
        cfg,
        target_range_m=target_range_m,
        range_gate_bins=range_gate_bins,
        range_side=range_side,
    )
    gated = range_cube[:, :, :, start:end]
    if max_frames is not None:
        gated = gated[: max(0, int(max_frames))]
    if gated.shape[0] == 0:
        raise RuntimeError(f"Complex coherent extraction produced zero frames: {capture_path}")
    if gated.shape[-1] == 0:
        raise RuntimeError(
            "Complex coherent extraction produced zero range bins after gate "
            f"[{start}:{end}] for capture {capture_path}"
        )

    coherent = np.mean(gated, axis=1).astype(np.complex64, copy=False)
    return np.stack([coherent.real, coherent.imag], axis=-1).astype(np.float32, copy=False)


def extract_frame_zero_doppler_by_rx_db(
    capture_path: Path,
    cfg: RadarTensorConfig,
    *,
    range_side: str,
    window_kind: str,
    eps: float,
    max_frames: int | None,
) -> np.ndarray:
    """Return per-frame zero-Doppler power as dB with shape `[F, RX, B]`."""
    processed = _process_capture_lenient(
        capture_path,
        cfg,
        window_kind=window_kind,
        eps=eps,
    )
    return _extract_zero_doppler_by_rx_db_from_processed(
        processed,
        range_side=range_side,
        eps=eps,
        max_frames=max_frames,
    )


def _build_range_bin_mask(
    cfg: RadarTensorConfig,
    *,
    range_side: str,
    range_min_m: float | None,
    range_max_m: float | None,
) -> np.ndarray | None:
    """Build optional range-bin mask for distance-gated feature extraction."""
    if range_min_m is None and range_max_m is None:
        return None
    if range_min_m is not None and range_min_m < 0.0:
        raise ValueError(f"range_min_m must be >= 0, got {range_min_m}")
    if range_max_m is not None and range_max_m < 0.0:
        raise ValueError(f"range_max_m must be >= 0, got {range_max_m}")
    if (
        range_min_m is not None
        and range_max_m is not None
        and float(range_min_m) > float(range_max_m)
    ):
        raise ValueError(
            "range_min_m must be <= range_max_m when both are set, "
            f"got min={range_min_m}, max={range_max_m}"
        )

    axis_m = compute_range_axis_m(cfg)
    if range_side != "full":
        axis_m = select_useful_range_side(axis_m, side=range_side)
    distances_m = np.abs(np.asarray(axis_m, dtype=np.float32))

    lo = 0.0 if range_min_m is None else float(range_min_m)
    hi = np.inf if range_max_m is None else float(range_max_m)
    mask = (distances_m >= lo) & (distances_m <= hi)
    if not np.any(mask):
        raise RuntimeError(
            "Range gate removed all bins. "
            f"Requested [{lo}, {hi}] m with range_side={range_side!r}."
        )
    return mask


def load_baseline_zero_doppler_mean_by_rx_db(
    capture_path: Path,
    cfg: RadarTensorConfig,
    *,
    range_side: str,
    window_kind: str,
    eps: float,
    range_min_m: float | None = None,
    range_max_m: float | None = None,
) -> np.ndarray:
    """Load one baseline capture and return mean zero-Doppler dB as `[RX, B]`."""
    baseline_frames = extract_frame_zero_doppler_by_rx_db(
        capture_path,
        cfg,
        range_side=range_side,
        window_kind=window_kind,
        eps=eps,
        max_frames=None,
    )
    range_mask = _build_range_bin_mask(
        cfg,
        range_side=range_side,
        range_min_m=range_min_m,
        range_max_m=range_max_m,
    )
    if range_mask is not None:
        baseline_frames = baseline_frames[:, :, range_mask]
    if baseline_frames.shape[0] == 0:
        raise RuntimeError(f"Baseline capture produced zero frames: {capture_path}")
    return np.mean(baseline_frames, axis=0, dtype=np.float32).astype(np.float32, copy=False)


def extract_frame_feature_tensor_by_rx_db(
    capture_path: Path,
    cfg: RadarTensorConfig,
    *,
    range_side: str,
    window_kind: str,
    eps: float,
    max_frames: int | None,
    baseline_open_mean_db: np.ndarray | None = None,
    baseline_blocked_mean_db: np.ndarray | None = None,
    range_min_m: float | None = None,
    range_max_m: float | None = None,
    feature_mode: str = FEATURE_MODE_ZERO_DOPPLER_DB,
    target_range_m: float = 0.40,
    range_gate_bins: int = 2,
    background_subtraction: str = BACKGROUND_SUBTRACTION_NONE,
    background_reference: BackgroundReference | None = None,
) -> np.ndarray:
    """Return frame features according to selected feature mode.

    `zero_doppler_db`:
      - no baselines: raw zero-Doppler dB `[F, RX, B]`
      - one baseline: delta dB `[F, RX, B]`
      - two baselines: concatenated deltas `[F, RX, 2*B]` (open then blocked)

    `complex_coherent`:
      - coherent complex range-FFT average with Re/Im channels `[F, RX, B, 2]`
    """
    if feature_mode == FEATURE_MODE_COMPLEX_COHERENT:
        if baseline_open_mean_db is not None or baseline_blocked_mean_db is not None:
            raise ValueError(
                "Baseline dB deltas are only supported for feature_mode='zero_doppler_db'. "
                "Use background_subtraction with --background-capture for complex mode."
            )
        return extract_frame_complex_coherent_by_rx_reim(
            capture_path,
            cfg,
            range_side=range_side,
            window_kind=window_kind,
            eps=eps,
            max_frames=max_frames,
            target_range_m=target_range_m,
            range_gate_bins=range_gate_bins,
            background_subtraction=background_subtraction,
            background_reference=background_reference,
        )

    if feature_mode != FEATURE_MODE_ZERO_DOPPLER_DB:
        raise ValueError(
            "Unsupported feature_mode. "
            f"Use '{FEATURE_MODE_ZERO_DOPPLER_DB}' or '{FEATURE_MODE_COMPLEX_COHERENT}', "
            f"got {feature_mode!r}."
        )

    frame_db = extract_frame_zero_doppler_by_rx_db(
        capture_path,
        cfg,
        range_side=range_side,
        window_kind=window_kind,
        eps=eps,
        max_frames=max_frames,
    )
    range_mask = _build_range_bin_mask(
        cfg,
        range_side=range_side,
        range_min_m=range_min_m,
        range_max_m=range_max_m,
    )
    if range_mask is not None:
        frame_db = frame_db[:, :, range_mask]

    target_shape = frame_db.shape[1:]
    deltas: list[np.ndarray] = []

    if baseline_open_mean_db is not None:
        open_mean = np.asarray(baseline_open_mean_db, dtype=np.float32)
        if open_mean.shape != target_shape:
            raise ValueError(
                "Open baseline shape does not match frame feature shape. "
                f"baseline={open_mean.shape} expected={target_shape}"
            )
        deltas.append((frame_db - open_mean[None, ...]).astype(np.float32, copy=False))

    if baseline_blocked_mean_db is not None:
        blocked_mean = np.asarray(baseline_blocked_mean_db, dtype=np.float32)
        if blocked_mean.shape != target_shape:
            raise ValueError(
                "Blocked baseline shape does not match frame feature shape. "
                f"baseline={blocked_mean.shape} expected={target_shape}"
            )
        deltas.append((frame_db - blocked_mean[None, ...]).astype(np.float32, copy=False))

    if not deltas:
        return frame_db
    if len(deltas) == 1:
        return deltas[0]
    return np.concatenate(deltas, axis=2).astype(np.float32, copy=False)


def flatten_frame_features(frames_rx_bins: np.ndarray) -> np.ndarray:
    """Flatten `[N, RX, B]` features into `[N, RX*B]`."""
    arr = np.asarray(frames_rx_bins, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [N,RX,B] feature tensor, got shape {arr.shape}")
    return arr.reshape(arr.shape[0], -1).astype(np.float32, copy=False)


def feature_tensor_to_samples(frames_rx_bins: np.ndarray, *, sample_mode: str) -> np.ndarray:
    """Convert per-frame features into model samples.

    Supports:
    - `[F,RX,B]` real-valued tensors
    - `[F,RX,B,2]` Re/Im tensors from `complex_coherent`
    """
    arr = np.asarray(frames_rx_bins, dtype=np.float32)
    if arr.ndim not in {3, 4}:
        raise ValueError(f"Expected [F,RX,B] or [F,RX,B,2], got shape {arr.shape}")
    if arr.shape[0] == 0:
        if arr.ndim == 3:
            return np.zeros((0, arr.shape[1] * arr.shape[2]), dtype=np.float32)
        return np.zeros((0, arr.shape[1] * arr.shape[2] * arr.shape[3]), dtype=np.float32)

    if arr.ndim == 3:
        arr_flat = arr.reshape(arr.shape[0], -1).astype(np.float32, copy=False)
    else:
        if arr.shape[-1] != 2:
            raise ValueError(f"Expected last dim size 2 for Re/Im channels, got {arr.shape}")
        arr_flat = arr.reshape(arr.shape[0], -1).astype(np.float32, copy=False)

    if sample_mode == "frames":
        return arr_flat

    if sample_mode == "capture-mean":
        return np.mean(arr_flat, axis=0, keepdims=True, dtype=np.float32).astype(
            np.float32, copy=False
        )

    if sample_mode == "capture-mean-std":
        mean = np.mean(arr_flat, axis=0, dtype=np.float32).reshape(1, -1)
        std = np.std(arr_flat, axis=0, dtype=np.float32).reshape(1, -1)
        return np.concatenate([mean, std], axis=1).astype(np.float32, copy=False)

    raise ValueError(
        "Unsupported sample_mode. "
        "Use one of: 'frames', 'capture-mean', 'capture-mean-std'. "
        f"Got {sample_mode!r}."
    )


def standardize_features(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features using train split statistics."""
    mean = np.mean(train_x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(train_x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, float(eps))

    train_norm = ((train_x - mean) / std).astype(np.float32)
    val_norm = ((val_x - mean) / std).astype(np.float32)
    test_norm = ((test_x - mean) / std).astype(np.float32)
    return train_norm, val_norm, test_norm, mean, std


def per_recording_standardize_features(
    x: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Standardize each sample vector independently."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [N,D] feature matrix, got shape {arr.shape}")
    mean = np.mean(arr, axis=1, keepdims=True, dtype=np.float32)
    std = np.std(arr, axis=1, keepdims=True, dtype=np.float32)
    std = np.maximum(std, float(eps))
    return ((arr - mean) / std).astype(np.float32, copy=False)


def apply_normalization_mode(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    *,
    normalization_mode: str,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply selected normalization strategy and return data + train stats."""
    mode = str(normalization_mode)
    if mode == NORMALIZATION_TRAINSET:
        return standardize_features(train_x, val_x, test_x, eps=eps)
    if mode == NORMALIZATION_PER_RECORDING:
        train_norm = per_recording_standardize_features(train_x, eps=eps)
        val_norm = per_recording_standardize_features(val_x, eps=eps)
        test_norm = per_recording_standardize_features(test_x, eps=eps)
        dim = int(train_x.shape[1])
        mean = np.zeros((dim,), dtype=np.float32)
        std = np.ones((dim,), dtype=np.float32)
        return train_norm, val_norm, test_norm, mean, std
    if mode == NORMALIZATION_NONE:
        dim = int(train_x.shape[1])
        mean = np.zeros((dim,), dtype=np.float32)
        std = np.ones((dim,), dtype=np.float32)
        return (
            np.asarray(train_x, dtype=np.float32),
            np.asarray(val_x, dtype=np.float32),
            np.asarray(test_x, dtype=np.float32),
            mean,
            std,
        )
    raise ValueError(
        "Unsupported normalization_mode. "
        f"Use one of '{NORMALIZATION_NONE}', '{NORMALIZATION_PER_RECORDING}', "
        f"'{NORMALIZATION_TRAINSET}'. Got {mode!r}."
    )


def resolve_regression_target(
    capture: LabeledCapture,
    *,
    label_target_map: dict[str, float] | None,
    numeric_target_regex: str,
) -> float:
    """Resolve regression target from map or parsed label/file stem."""
    if label_target_map is not None and capture.label in label_target_map:
        return float(label_target_map[capture.label])

    pattern = _compile_pattern(numeric_target_regex, "numeric-target")
    for raw in (capture.label, capture.path.stem):
        match = pattern.search(raw)
        if match is None:
            continue
        token = match.group(1) if match.lastindex else match.group(0)
        return float(token)

    raise RuntimeError(
        "Could not resolve regression target for capture "
        f"{capture.path.name!r} with label {capture.label!r}. "
        "Provide --label-target-map or adjust --numeric-target-regex."
    )


def load_label_target_map(path: Path | None) -> dict[str, float] | None:
    """Load optional JSON mapping from label to regression target."""
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in label-target map JSON, got {type(payload).__name__}")

    out: dict[str, float] = {}
    for key, value in payload.items():
        out[str(key)] = float(value)
    return out


def compute_average_range_doppler_by_rx(
    processed: ProcessedRangeTensors,
    *,
    range_side: str,
) -> np.ndarray:
    """Compute average range-Doppler power by receiver as `[RX, B, D]`."""
    power = np.abs(processed.doppler_cube) ** 2
    avg_chirp_major = np.mean(power, axis=0).astype(np.float32)  # [D, RX, B]
    rx_range_doppler = np.transpose(avg_chirp_major, (1, 2, 0))  # [RX, B, D]

    if range_side != "full":
        rx_range_doppler = select_useful_range_side(rx_range_doppler, side=range_side)

    return rx_range_doppler.astype(np.float32, copy=False)


def compute_average_range_profile_by_rx(range_doppler_by_rx: np.ndarray) -> np.ndarray:
    """Average over Doppler bins to produce `[RX, B]` range profiles."""
    arr = np.asarray(range_doppler_by_rx, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [RX,B,D], got {arr.shape}")
    return np.mean(arr, axis=2).astype(np.float32)
