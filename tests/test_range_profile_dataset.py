"""Tests for range-profile dataset preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmwave_radar_devtool.range_profile_dataset import (
    RadarTensorConfig,
    create_training_inputs,
    decode_complex_xwr18xx,
    process_capture,
    reshape_to_cube,
)


def test_decode_complex_xwr18xx_iiqq_grouping() -> None:
    """Complex decoder should map I0 I1 Q0 Q1 groups as documented."""
    words = np.array([10, 20, 1, 2, 30, 40, 3, 4], dtype=np.int16)
    decoded = decode_complex_xwr18xx(words)
    expected = np.array([10 + 1j, 20 + 2j, 30 + 3j, 40 + 4j], dtype=np.complex64)
    np.testing.assert_allclose(decoded, expected)


def test_reshape_to_cube_shape() -> None:
    """Reshape should produce [frame, chirp, rx, adc_sample]."""
    cfg = RadarTensorConfig(
        chirps_per_frame=2,
        num_rx=2,
        num_tx=1,
        num_adc_samples=4,
        adc_sample_rate_ksps=6000.0,
        freq_slope_mhz_per_us=80.0,
        start_freq_ghz=77.0,
        idle_time_us=10.0,
        ramp_end_time_us=50.0,
    )
    complex_stream = np.arange(2 * 2 * 2 * 4, dtype=np.float32).astype(np.complex64)
    cube = reshape_to_cube(complex_stream, cfg)
    assert cube.shape == (2, 2, 2, 4)


def test_process_capture_outputs_expected_shapes(tmp_path: Path) -> None:
    """Processing should emit tensors with consistent shapes."""
    cfg = RadarTensorConfig(
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
    num_frames = 3
    # Each complex sample consumes two int16 words, but decoder expects iiqq groups.
    # Build words from grouped [I0, I1, Q0, Q1] tuples.
    num_complex = num_frames * cfg.complex_samples_per_frame
    i_vals = np.arange(num_complex, dtype=np.int16)
    q_vals = (i_vals + 100).astype(np.int16)
    groups = np.column_stack(
        [
            i_vals[0::2],
            i_vals[1::2],
            q_vals[0::2],
            q_vals[1::2],
        ]
    ).reshape(-1)
    capture_path = tmp_path / "object.bin"
    groups.astype("<i2").tofile(capture_path)

    out = process_capture(capture_path, cfg)
    assert out.cube.shape == (num_frames, 2, 2, 8)
    assert out.range_cube.shape == (num_frames, 2, 2, 8)
    assert np.iscomplexobj(out.range_cube)
    assert out.log_magnitude_db.shape == (num_frames, 2, 2, 8)
    assert out.zero_doppler_db.shape == (num_frames, 8)
    assert out.range_doppler_power_unshifted.shape == (num_frames, 8, 2)
    assert out.range_doppler_power_shifted.shape == (num_frames, 8, 2)


def test_process_capture_can_trim_trailing_partial_frame(tmp_path: Path) -> None:
    """Optional trimming should keep full frames and drop a partial capture tail."""
    cfg = RadarTensorConfig(
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
    num_complex = cfg.complex_samples_per_frame + 2
    i_vals = np.arange(num_complex, dtype=np.int16)
    q_vals = (i_vals + 100).astype(np.int16)
    groups = np.column_stack(
        [
            i_vals[0::2],
            i_vals[1::2],
            q_vals[0::2],
            q_vals[1::2],
        ]
    ).reshape(-1)
    capture_path = tmp_path / "partial_tail.bin"
    groups.astype("<i2").tofile(capture_path)

    with pytest.raises(ValueError, match="not divisible"):
        process_capture(capture_path, cfg)

    with pytest.warns(RuntimeWarning, match="trailing partial frame"):
        out = process_capture(capture_path, cfg, allow_trailing_partial_frame=True)

    assert out.cube.shape == (1, 2, 2, 8)


def test_process_capture_partial_trim_warning_contains_capture_and_percentage(
    tmp_path: Path,
) -> None:
    """Trailing-frame warning should include capture name and drop percentage."""
    cfg = RadarTensorConfig(
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
    num_complex = cfg.complex_samples_per_frame + 2
    i_vals = np.arange(num_complex, dtype=np.int16)
    q_vals = (i_vals + 100).astype(np.int16)
    groups = np.column_stack(
        [
            i_vals[0::2],
            i_vals[1::2],
            q_vals[0::2],
            q_vals[1::2],
        ]
    ).reshape(-1)
    capture_path = tmp_path / "warn_details.bin"
    groups.astype("<i2").tofile(capture_path)

    with pytest.warns(RuntimeWarning, match=r"capture=.*warn_details.bin.*% of total"):
        process_capture(capture_path, cfg, allow_trailing_partial_frame=True)


def test_process_capture_rejects_large_partial_frame_drop(tmp_path: Path) -> None:
    """Large trailing drops should be rejected instead of silently trimmed."""
    cfg = RadarTensorConfig(
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
    # One full frame + 20 extra complex samples -> 20/148 ~= 13.5% drop.
    num_complex = cfg.complex_samples_per_frame + 20
    i_vals = np.arange(num_complex, dtype=np.int16)
    q_vals = (i_vals + 100).astype(np.int16)
    groups = np.column_stack(
        [
            i_vals[0::2],
            i_vals[1::2],
            q_vals[0::2],
            q_vals[1::2],
        ]
    ).reshape(-1)
    capture_path = tmp_path / "large_drop.bin"
    groups.astype("<i2").tofile(capture_path)

    with pytest.raises(ValueError, match="large trailing partial frame"):
        process_capture(capture_path, cfg, allow_trailing_partial_frame=True)


def test_create_training_inputs_windowing_and_background() -> None:
    """Training input helper should support background subtraction and sliding windows."""
    object_logmag = np.ones((5, 2, 2, 4), dtype=np.float32) * 7.0
    empty_logmag = np.ones((3, 2, 2, 4), dtype=np.float32) * 2.0
    windows = create_training_inputs(
        object_logmag,
        empty_logmag=empty_logmag,
        window_frames=3,
        window_step=1,
    )
    assert windows.shape == (3, 3, 2, 2, 4)
    assert np.allclose(windows, 5.0)


def test_decode_rejects_non_multiple_of_four() -> None:
    """Decoder should enforce 4-word grouping validation."""
    words = np.array([1, 2, 3], dtype=np.int16)
    with pytest.raises(ValueError):
        decode_complex_xwr18xx(words)
