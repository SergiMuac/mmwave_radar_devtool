"""Tests for supported board profiles."""

import pytest

from mmwave_radar_devtool.exceptions import ConfigurationError
from mmwave_radar_devtool.profiles import get_profile, list_profiles


def test_list_profiles_includes_supported_boards() -> None:
    """The package should expose both concrete TI board profiles."""
    names = {profile.name for profile in list_profiles()}
    assert "generic-ti-dca1000" in names
    assert "iwr1843-dca1000" in names
    assert "iwr6843-dca1000" in names
    assert "iwr6843-usb" in names
    assert "iwr6843aop-dca1000" in names
    assert "iwr6843aop-usb" in names


def test_get_profile_returns_iwr6843aop_profile() -> None:
    """The new IWR6843AOP profile should be discoverable."""
    profile = get_profile("iwr6843aop-dca1000")
    assert profile.display_name == "IWR6843AOP + DCA1000"
    assert profile.radar_family == "xwr68xx"


def test_get_profile_rejects_unknown_name() -> None:
    """Unknown profile names should produce a useful error."""
    with pytest.raises(ConfigurationError):
        get_profile("does-not-exist")


def test_usb_profile_exposes_telemetry_capabilities() -> None:
    """The USB profile should advertise telemetry-only support."""
    profile = get_profile("iwr6843-usb")
    assert profile.supports_raw_capture is False
    assert profile.supports_usb_telemetry is True
    assert profile.data_backend == "ti-data-uart"


def test_iwr6843_usb_profile_points_to_demo_cfg() -> None:
    """The non-AOP USB profile should default to the uploaded xwr68xx config."""
    profile = get_profile("iwr6843-usb")
    assert str(profile.default_cfg_path) == "config/xwr68xx_profile_2026_04_02T12_19_57_603.cfg"
