"""Tests for live dashboard rendering helpers."""

import numpy as np
import pytest
from rich.panel import Panel

from mmwave_radar_devtool.cfg_parser import parse_radar_cfg
from mmwave_radar_devtool.config import CaptureConfig
from mmwave_radar_devtool.live_view import (
    LivePredictionResult,
    LiveSignalProcessingConfig,
    PlotSeries,
    SignalViewMode,
    TerminalLiveDashboard,
    _decode_dca_complex_words,
    _magnitude_to_db,
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
    assert plot_series.title.startswith("Zero-Doppler range profile")


def test_dashboard_range_mode_supports_baseline_delta_view() -> None:
    """Range mode should expose delta title when baseline profile is provided."""
    processing = LiveSignalProcessingConfig(
        baseline_range_db=np.zeros(4, dtype=np.float32),
    )
    dashboard = TerminalLiveDashboard(processing_config=processing)
    dashboard.metrics.current_mode = SignalViewMode.RANGE
    plot_series = dashboard._build_plot_series()
    assert "absolute range delta vs baseline" in plot_series.title
    assert plot_series.values.size == 4
    assert plot_series.y_min is not None
    assert plot_series.y_min == pytest.approx(0.0)


def test_dashboard_range_mode_uses_configured_baseline_delta_limits() -> None:
    """Baseline delta plots should honor explicit display limits."""
    processing = LiveSignalProcessingConfig(
        baseline_range_db=np.zeros(4, dtype=np.float32),
        range_db_min=0.0,
        range_db_max=120.0,
    )
    dashboard = TerminalLiveDashboard(processing_config=processing)
    dashboard.metrics.current_mode = SignalViewMode.RANGE
    plot_series = dashboard._build_plot_series()
    assert plot_series.y_min == pytest.approx(0.0)
    assert plot_series.y_max == pytest.approx(120.0)


def test_resolve_db_limits_respects_fixed_configured_bounds() -> None:
    """Configured min/max should remain fixed and not auto-expand."""
    processing = LiveSignalProcessingConfig(
        spectrum_db_min=0.0,
        spectrum_db_max=120.0,
        range_db_min=0.0,
        range_db_max=120.0,
    )
    dashboard = TerminalLiveDashboard(processing_config=processing)

    spectrum_min, spectrum_max = dashboard._resolve_db_limits(
        SignalViewMode.SPECTRUM,
        np.array([30.0, 150.0, 200.0], dtype=np.float32),
        configured_min=processing.spectrum_db_min,
        configured_max=processing.spectrum_db_max,
    )
    range_min, range_max = dashboard._resolve_db_limits(
        SignalViewMode.RANGE,
        np.array([20.0, 140.0, 220.0], dtype=np.float32),
        configured_min=processing.range_db_min,
        configured_max=processing.range_db_max,
    )

    assert spectrum_min == pytest.approx(0.0)
    assert spectrum_max == pytest.approx(120.0)
    assert range_min == pytest.approx(0.0)
    assert range_max == pytest.approx(120.0)


def test_dashboard_live_prediction_callback_receives_latest_frame() -> None:
    """Live prediction should run on `[RX,B]` zero-Doppler dB frames."""
    cfg = parse_radar_cfg("config/xwr18xx_profile_raw_capture.cfg")
    seen_shapes: list[tuple[int, ...]] = []

    def _fake_predict(frame_db: np.ndarray) -> LivePredictionResult:
        seen_shapes.append(tuple(frame_db.shape))
        return LivePredictionResult(
            task="classification",
            primary="none",
            confidence=0.99,
            detail="fake",
        )

    processing = LiveSignalProcessingConfig(
        prediction_callback=_fake_predict,
        prediction_interval_s=0.0,
    )
    dashboard = TerminalLiveDashboard(radar_cfg=cfg, processing_config=processing)
    assert dashboard._profile_cfg is not None
    samples_per_chirp = dashboard._profile_cfg.num_adc_samples
    samples_per_frame = dashboard._chirps_per_frame * dashboard._rx_count * samples_per_chirp
    dashboard.metrics.complex_history = np.ones(samples_per_frame, dtype=np.complex64)

    prediction = dashboard._update_prediction()

    assert prediction is not None
    assert prediction.primary == "none"
    assert seen_shapes == [(dashboard._rx_count, max(8, samples_per_chirp // 2))]


def test_decode_dca_complex_words_handles_iiqq_order() -> None:
    """DCA words should decode from IIQQ grouping into complex samples."""
    words = np.array([10, 20, 1, 2, 30, 40, 3, 4], dtype=np.float32)
    decoded = _decode_dca_complex_words(words)
    expected = np.array([10 + 1j, 20 + 2j, 30 + 3j, 40 + 4j], dtype=np.complex64)
    np.testing.assert_allclose(decoded, expected)


def test_decode_dca_complex_words_supports_q_first_streams() -> None:
    """Decoder should support Q-first ordering when sampleSwap indicates it."""
    words = np.array([1, 10, 2, 20, 3, 30, 4, 40], dtype=np.float32)
    decoded = _decode_dca_complex_words(words, decode_order="iqiq", q_first=True)
    expected = np.array([10 + 1j, 20 + 2j, 30 + 3j, 40 + 4j], dtype=np.complex64)
    np.testing.assert_allclose(decoded, expected)


def test_chirp_aligned_spectrum_uses_cfg_sample_count() -> None:
    """Range FFT size should follow chirp sample count from cfg context."""
    cfg = parse_radar_cfg("config/xwr18xx_profile_raw_capture.cfg")
    dashboard = TerminalLiveDashboard(radar_cfg=cfg)
    assert dashboard._profile_cfg is not None
    samples_per_chirp = dashboard._profile_cfg.num_adc_samples
    chirp_block = samples_per_chirp * dashboard._rx_count
    dashboard.metrics.latest_complex = np.ones(chirp_block * 2, dtype=np.complex64)
    dashboard.metrics.complex_history = dashboard.metrics.latest_complex.copy()

    spectrum = dashboard._compute_chirp_aligned_spectrum()

    assert spectrum.size == max(8, samples_per_chirp // 2)


def test_magnitude_to_db_peak_normalization_and_floor() -> None:
    """Relative dB conversion should pin peak to 0 dB and honor floor."""
    values = np.array([100.0, 10.0, 0.0], dtype=np.float32)
    db = _magnitude_to_db(values, normalize_to_peak=True, floor_db=-70.0)

    assert db[0] == pytest.approx(0.0, abs=1e-4)
    assert db[1] == pytest.approx(-20.0, abs=1e-3)
    assert db[2] == pytest.approx(-70.0, abs=1e-4)
