"""Tests for cfg parsing."""

from pathlib import Path

import pytest

from mmwave_radar_devtool.cfg_parser import create_capture_cfg, parse_radar_cfg
from mmwave_radar_devtool.config import DCA1000DataLoggingMode
from mmwave_radar_devtool.exceptions import ConfigurationError


def test_parse_cfg_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    """The parser should keep only CLI commands."""
    cfg_path = tmp_path / "test.cfg"
    cfg_path.write_text("% comment\n\nsensorStop\nprofileCfg 1 2 3\n", encoding="ascii")

    cfg = parse_radar_cfg(cfg_path)

    assert cfg.texts() == ["sensorStop", "profileCfg 1 2 3"]


def test_parse_cfg_rejects_non_ascii_content(tmp_path: Path) -> None:
    """The parser should reject non-ASCII cfg content."""
    cfg_path = tmp_path / "bad.cfg"
    cfg_path.write_text("sensorStop\nprófileCfg 1 2 3\n", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        parse_radar_cfg(cfg_path)


def test_validate_capture_cfg_extracts_adc_and_lvds_requirements(tmp_path: Path) -> None:
    """The parser should derive DCA1000 capture settings from cfg commands."""
    cfg_path = tmp_path / "capture.cfg"
    cfg_path.write_text(
        "sensorStop\nflushCfg\nadcCfg 2 1\nlvdsStreamCfg -1 0 1 0\nsensorStart\n",
        encoding="ascii",
    )

    cfg = parse_radar_cfg(cfg_path)
    requirements = cfg.validate_for_dca_capture()

    assert requirements.data_format_mode == 3
    assert requirements.data_logging_mode is DCA1000DataLoggingMode.RAW


def test_validate_capture_cfg_rejects_missing_hw_stream(tmp_path: Path) -> None:
    """The validator should reject cfg files without hardware LVDS streaming."""
    cfg_path = tmp_path / "bad_capture.cfg"
    cfg_path.write_text(
        "sensorStop\nflushCfg\nadcCfg 2 1\nlvdsStreamCfg -1 0 0 0\nsensorStart\n",
        encoding="ascii",
    )

    cfg = parse_radar_cfg(cfg_path)

    with pytest.raises(ConfigurationError):
        cfg.validate_for_dca_capture()


def test_create_capture_cfg_rewrites_lvds_stream_cfg(tmp_path: Path) -> None:
    """The capture cfg helper should rewrite lvdsStreamCfg in place."""
    source_path = tmp_path / "input.cfg"
    target_path = tmp_path / "output.cfg"
    source_path.write_text("adcCfg 2 1\nlvdsStreamCfg -1 0 0 0\nsensorStart\n", encoding="ascii")

    create_capture_cfg(
        source_path=source_path,
        target_path=target_path,
        enable_header=False,
        enable_hw_stream=True,
        enable_sw_stream=False,
    )

    result = target_path.read_text(encoding="ascii")
    assert "lvdsStreamCfg -1 0 1 0" in result
