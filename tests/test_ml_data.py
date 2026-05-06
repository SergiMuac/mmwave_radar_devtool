"""Tests for ML data helper utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mmwave_radar_devtool.ml.data as ml_data
from mmwave_radar_devtool.ml.data import (
    BACKGROUND_SUBTRACTION_COMPLEX_RANGE,
    FEATURE_MODE_ZERO_DOPPLER_DB,
    LabeledCapture,
    compute_chirp_coherence,
    compute_target_range_gate_slice,
    discover_labeled_captures,
    extract_frame_complex_coherent_by_rx_reim,
    extract_frame_feature_tensor_by_rx_db,
    feature_tensor_to_samples,
    load_background_reference,
    load_baseline_zero_doppler_mean_by_rx_db,
    parse_hidden_dims,
    per_recording_standardize_features,
    resolve_regression_target,
    split_captures_stratified,
)
from mmwave_radar_devtool.range_profile_dataset import ProcessedRangeTensors, RadarTensorConfig


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x01")


def _dummy_cfg() -> RadarTensorConfig:
    return RadarTensorConfig(
        chirps_per_frame=2,
        num_rx=2,
        num_tx=1,
        num_adc_samples=8,
        adc_sample_rate_ksps=6000.0,
        freq_slope_mhz_per_us=80.0,
        start_freq_ghz=77.0,
        idle_time_us=10.0,
        ramp_end_time_us=50.0,
    )


def _fake_processed(
    *,
    range_cube: np.ndarray,
    cube: np.ndarray | None = None,
) -> ProcessedRangeTensors:
    range_cube_arr = np.asarray(range_cube, dtype=np.complex64)
    if cube is None:
        cube = np.zeros_like(range_cube_arr, dtype=np.complex64)
    cube_arr = np.asarray(cube, dtype=np.complex64)
    doppler_cube = np.fft.fft(range_cube_arr, axis=1).astype(np.complex64)
    magnitude = np.abs(range_cube_arr).astype(np.float32)
    power = (magnitude**2).astype(np.float32)
    zero_doppler_power = np.mean(np.abs(doppler_cube[:, 0, :, :]) ** 2, axis=1).astype(np.float32)
    zero_doppler_db = (10.0 * np.log10(zero_doppler_power + 1e-9)).astype(np.float32)
    rd_power_chirp_major = np.mean(np.abs(doppler_cube) ** 2, axis=2).astype(np.float32)
    rd_unshifted = np.transpose(rd_power_chirp_major, (0, 2, 1))
    rd_shifted = np.fft.fftshift(rd_unshifted, axes=2).astype(np.float32)
    return ProcessedRangeTensors(
        cube=cube_arr,
        range_cube=range_cube_arr,
        magnitude=magnitude,
        power=power,
        log_magnitude_db=(20.0 * np.log10(magnitude + 1e-9)).astype(np.float32),
        log_power_db=(10.0 * np.log10(power + 1e-9)).astype(np.float32),
        doppler_cube=doppler_cube,
        zero_doppler_power=zero_doppler_power,
        zero_doppler_db=zero_doppler_db,
        range_doppler_power_unshifted=rd_unshifted.astype(np.float32),
        range_doppler_power_shifted=rd_shifted,
    )


def test_discover_labeled_captures_from_nested_folders(tmp_path: Path) -> None:
    """Nested folder names should map to capture labels."""
    _touch(tmp_path / "person" / "a.bin")
    _touch(tmp_path / "empty" / "b.bin")

    captures = discover_labeled_captures(tmp_path)
    labels = sorted(capture.label for capture in captures)

    assert labels == ["empty", "person"]


def test_discover_labeled_captures_from_flat_names(tmp_path: Path) -> None:
    """Flat `label_index.bin` names should parse labels by regex."""
    _touch(tmp_path / "person_0.bin")
    _touch(tmp_path / "person_1.bin")
    _touch(tmp_path / "empty_0.bin")

    captures = discover_labeled_captures(tmp_path)
    labels = sorted(capture.label for capture in captures)

    assert labels == ["empty", "person", "person"]


def test_discover_labeled_captures_reports_unlabeled_files(tmp_path: Path) -> None:
    """Files that do not match the label regex should raise an informative error."""
    _touch(tmp_path / "capture.bin")

    with pytest.raises(RuntimeError):
        discover_labeled_captures(tmp_path)


def test_split_captures_stratified_keeps_all_classes() -> None:
    """Stratified split should include each class in training when possible."""
    captures = [
        LabeledCapture(path=Path(f"a_{idx}.bin"), label="a") for idx in range(4)
    ] + [
        LabeledCapture(path=Path(f"b_{idx}.bin"), label="b") for idx in range(4)
    ]

    split = split_captures_stratified(
        captures,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )

    train_labels = {capture.label for capture in split.train}
    assert train_labels == {"a", "b"}
    assert len(split.val) > 0
    assert len(split.test) > 0


def test_resolve_regression_target_from_map_then_regex() -> None:
    """Regression target should prefer explicit map and fallback to regex."""
    capture_a = LabeledCapture(path=Path("distance_foo_0.bin"), label="distance_foo")
    capture_b = LabeledCapture(path=Path("sugar_35g_2.bin"), label="sugar_35g")

    target_a = resolve_regression_target(
        capture_a,
        label_target_map={"distance_foo": 1.23},
        numeric_target_regex=r"(\d+)",
    )
    target_b = resolve_regression_target(
        capture_b,
        label_target_map=None,
        numeric_target_regex=r"([-+]?\d+(?:\.\d+)?)",
    )

    assert target_a == pytest.approx(1.23)
    assert target_b == pytest.approx(35.0)


def test_parse_hidden_dims_rejects_empty() -> None:
    """Hidden-dimension parser should enforce at least one value."""
    with pytest.raises(ValueError):
        parse_hidden_dims(" , ")


def test_extract_frame_feature_tensor_applies_single_baseline_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single baseline should return frame-wise dB deltas with same shape."""
    frame_db = np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[11.0, 21.0], [31.0, 41.0]],
        ],
        dtype=np.float32,
    )

    def _fake_extract(*args: object, **kwargs: object) -> np.ndarray:
        return frame_db

    monkeypatch.setattr(ml_data, "extract_frame_zero_doppler_by_rx_db", _fake_extract)

    baseline_mean = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = extract_frame_feature_tensor_by_rx_db(
        Path("capture.bin"),
        object(),  # type: ignore[arg-type]
        range_side="positive",
        window_kind="hann",
        eps=1e-9,
        max_frames=None,
        baseline_open_mean_db=baseline_mean,
    )

    assert out.shape == frame_db.shape
    assert np.allclose(out, frame_db - baseline_mean[None, ...])


def test_extract_frame_feature_tensor_concatenates_dual_baseline_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dual baselines should concatenate open/block deltas along bin axis."""
    frame_db = np.array([[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)

    def _fake_extract(*args: object, **kwargs: object) -> np.ndarray:
        return frame_db

    monkeypatch.setattr(ml_data, "extract_frame_zero_doppler_by_rx_db", _fake_extract)

    open_mean = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    blocked_mean = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
    out = extract_frame_feature_tensor_by_rx_db(
        Path("capture.bin"),
        object(),  # type: ignore[arg-type]
        range_side="positive",
        window_kind="hann",
        eps=1e-9,
        max_frames=None,
        baseline_open_mean_db=open_mean,
        baseline_blocked_mean_db=blocked_mean,
    )

    expected_open = frame_db - open_mean[None, ...]
    expected_blocked = frame_db - blocked_mean[None, ...]
    expected = np.concatenate([expected_open, expected_blocked], axis=2)
    assert out.shape == expected.shape
    assert np.allclose(out, expected)


def test_load_baseline_zero_doppler_mean_by_rx_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Baseline loader should average frames into `[RX,B]` mean dB tensor."""
    baseline_frames = np.array(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[3.0, 5.0], [7.0, 9.0]],
        ],
        dtype=np.float32,
    )

    def _fake_extract(*args: object, **kwargs: object) -> np.ndarray:
        return baseline_frames

    monkeypatch.setattr(ml_data, "extract_frame_zero_doppler_by_rx_db", _fake_extract)

    out = load_baseline_zero_doppler_mean_by_rx_db(
        Path("baseline.bin"),
        object(),  # type: ignore[arg-type]
        range_side="positive",
        window_kind="hann",
        eps=1e-9,
    )

    assert out.shape == (2, 2)
    assert np.allclose(out, np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))


def test_extract_frame_feature_tensor_applies_range_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Range min/max should keep only bins inside requested distance window."""
    frame_db = np.array(
        [
            [[10.0, 20.0, 30.0, 40.0], [11.0, 21.0, 31.0, 41.0]],
        ],
        dtype=np.float32,
    )

    def _fake_extract(*args: object, **kwargs: object) -> np.ndarray:
        return frame_db

    monkeypatch.setattr(ml_data, "extract_frame_zero_doppler_by_rx_db", _fake_extract)
    monkeypatch.setattr(
        ml_data,
        "compute_range_axis_m",
        lambda cfg: np.array([0.1, 0.3, 0.6, 1.2], dtype=np.float32),
    )

    out = extract_frame_feature_tensor_by_rx_db(
        Path("capture.bin"),
        object(),  # type: ignore[arg-type]
        range_side="full",
        window_kind="hann",
        eps=1e-9,
        max_frames=None,
        range_min_m=0.25,
        range_max_m=0.7,
    )

    assert out.shape == (1, 2, 2)
    assert np.allclose(out[0, 0], np.array([20.0, 30.0], dtype=np.float32))


def test_load_baseline_zero_doppler_mean_by_rx_db_applies_range_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline mean should apply same range-window bin filtering."""
    baseline_frames = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
            [[5.0, 6.0, 7.0, 8.0], [50.0, 60.0, 70.0, 80.0]],
        ],
        dtype=np.float32,
    )

    def _fake_extract(*args: object, **kwargs: object) -> np.ndarray:
        return baseline_frames

    monkeypatch.setattr(ml_data, "extract_frame_zero_doppler_by_rx_db", _fake_extract)
    monkeypatch.setattr(
        ml_data,
        "compute_range_axis_m",
        lambda cfg: np.array([0.1, 0.3, 0.6, 1.2], dtype=np.float32),
    )

    out = load_baseline_zero_doppler_mean_by_rx_db(
        Path("baseline.bin"),
        object(),  # type: ignore[arg-type]
        range_side="full",
        window_kind="hann",
        eps=1e-9,
        range_min_m=0.25,
        range_max_m=0.7,
    )

    assert out.shape == (2, 2)
    assert np.allclose(out[0], np.array([4.0, 5.0], dtype=np.float32))
    assert np.allclose(out[1], np.array([40.0, 50.0], dtype=np.float32))


def test_feature_tensor_to_samples_supports_capture_aggregation() -> None:
    """Static captures can become one mean or mean/std sample instead of many frames."""
    frames = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[3.0, 4.0], [5.0, 6.0]],
        ],
        dtype=np.float32,
    )

    frame_samples = feature_tensor_to_samples(frames, sample_mode="frames")
    mean_sample = feature_tensor_to_samples(frames, sample_mode="capture-mean")
    mean_std_sample = feature_tensor_to_samples(frames, sample_mode="capture-mean-std")

    assert frame_samples.shape == (2, 4)
    assert np.allclose(frame_samples[0], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert mean_sample.shape == (1, 4)
    assert np.allclose(mean_sample[0], np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    assert mean_std_sample.shape == (1, 8)
    assert np.allclose(
        mean_std_sample[0],
        np.array([2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_feature_tensor_to_samples_accepts_complex_reim_last_axis() -> None:
    """Complex coherent `[F,RX,B,2]` tensors should flatten correctly."""
    frames = np.array(
        [
            [[[1.0, 0.5], [2.0, 1.5]], [[3.0, 2.5], [4.0, 3.5]]],
            [[[5.0, 4.5], [6.0, 5.5]], [[7.0, 6.5], [8.0, 7.5]]],
        ],
        dtype=np.float32,
    )
    samples = feature_tensor_to_samples(frames, sample_mode="frames")
    assert samples.shape == (2, 8)
    assert np.allclose(
        samples[0],
        np.array([1.0, 0.5, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5], dtype=np.float32),
    )


def test_extract_complex_coherent_output_shape_has_reim_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """complex_coherent should emit `[F,RX,B,2]` with preserved imag channel."""
    cfg = _dummy_cfg()
    range_cube = np.ones((2, 2, 2, 8), dtype=np.complex64) * (3.0 + 4.0j)

    def _fake_process(*args: object, **kwargs: object) -> ProcessedRangeTensors:
        return _fake_processed(range_cube=range_cube)

    monkeypatch.setattr(ml_data, "_process_capture_lenient", _fake_process)

    out = extract_frame_complex_coherent_by_rx_reim(
        Path("capture.bin"),
        cfg,
        range_side="positive",
        window_kind="hann",
        eps=1e-9,
        max_frames=None,
        target_range_m=0.4,
        range_gate_bins=1,
    )
    assert out.ndim == 4
    assert out.shape[0] == 2
    assert out.shape[1] == 2
    assert out.shape[-1] == 2


def test_extract_zero_doppler_mode_shape_unchanged_with_feature_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """zero_doppler_db mode should stay real-valued `[F,RX,B]`."""
    frame_db = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float32)

    def _fake_extract(*args: object, **kwargs: object) -> np.ndarray:
        return frame_db

    monkeypatch.setattr(ml_data, "extract_frame_zero_doppler_by_rx_db", _fake_extract)

    out = extract_frame_feature_tensor_by_rx_db(
        Path("capture.bin"),
        _dummy_cfg(),
        range_side="positive",
        window_kind="hann",
        eps=1e-9,
        max_frames=None,
        feature_mode=FEATURE_MODE_ZERO_DOPPLER_DB,
    )
    assert out.shape == (1, 2, 2)
    assert np.isrealobj(out)


def test_compute_target_range_gate_slice_bin_count_and_boundaries() -> None:
    """Target gate width should be 2*gate+1 unless truncated near edges."""
    cfg = _dummy_cfg()
    start, end, _, _ = compute_target_range_gate_slice(
        cfg,
        target_range_m=4.2,
        range_gate_bins=2,
        range_side="full",
    )
    assert end - start == 5

    start_edge, end_edge, _, _ = compute_target_range_gate_slice(
        cfg,
        target_range_m=0.0,
        range_gate_bins=2,
        range_side="positive",
    )
    assert end_edge - start_edge <= 5


def test_complex_range_subtraction_zero_when_object_equals_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If object equals empty recording, complex_range subtraction should be near zero."""
    cfg = _dummy_cfg()
    range_cube = np.ones((2, 2, 2, 8), dtype=np.complex64) * (5.0 - 2.0j)
    processed = _fake_processed(range_cube=range_cube)

    def _fake_process(*args: object, **kwargs: object) -> ProcessedRangeTensors:
        return processed

    monkeypatch.setattr(ml_data, "_process_capture_lenient", _fake_process)
    background = load_background_reference(
        Path("empty.bin"),
        cfg,
        window_kind="hann",
        eps=1e-9,
        background_subtraction=BACKGROUND_SUBTRACTION_COMPLEX_RANGE,
    )
    out = extract_frame_complex_coherent_by_rx_reim(
        Path("object.bin"),
        cfg,
        range_side="positive",
        window_kind="hann",
        eps=1e-9,
        max_frames=None,
        target_range_m=0.4,
        range_gate_bins=1,
        background_subtraction=BACKGROUND_SUBTRACTION_COMPLEX_RANGE,
        background_reference=background,
    )
    assert np.allclose(out, 0.0, atol=1e-6)


def test_split_captures_stratified_has_no_capture_leakage() -> None:
    """The same capture path should never appear in multiple splits."""
    captures = [
        LabeledCapture(path=Path(f"class_a_{idx}.bin"), label="a") for idx in range(5)
    ] + [
        LabeledCapture(path=Path(f"class_b_{idx}.bin"), label="b") for idx in range(5)
    ]
    split = split_captures_stratified(captures, val_ratio=0.2, test_ratio=0.2, seed=1)
    train_paths = {item.path for item in split.train}
    val_paths = {item.path for item in split.val}
    test_paths = {item.path for item in split.test}
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)


def test_compute_chirp_coherence_matches_definition() -> None:
    """Chirp coherence helper should return values in [0,1] for stable chirps."""
    complex_range = np.ones((2, 4, 2, 3), dtype=np.complex64) * (2.0 + 1.0j)
    coherence = compute_chirp_coherence(complex_range)
    assert coherence.shape == (2, 2, 3)
    assert np.all(coherence >= 0.0)
    assert np.all(coherence <= 1.0 + 1e-5)
    assert np.allclose(coherence, 1.0, atol=1e-5)


def test_per_recording_standardize_features() -> None:
    """Per-recording standardization should normalize each sample independently."""
    x = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]], dtype=np.float32)
    y = per_recording_standardize_features(x)
    assert y.shape == x.shape
    assert np.allclose(np.mean(y[0]), 0.0, atol=1e-6)
    assert np.allclose(np.std(y[0]), 1.0, atol=1e-6)
    assert np.allclose(y[1], 0.0, atol=1e-6)
