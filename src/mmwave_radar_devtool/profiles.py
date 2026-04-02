"""Known radar board and transport profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .exceptions import ConfigurationError


@dataclass(slots=True, frozen=True)
class RadarDeviceProfile:
    """Describes a supported radar board and data transport combination."""

    name: str
    display_name: str
    radar_family: str
    control_backend: str
    data_backend: str
    supports_raw_capture: bool
    supports_live_view: bool
    supports_usb_telemetry: bool
    description: str
    default_cfg_path: Path | None = None


_PROFILE_REGISTRY: dict[str, RadarDeviceProfile] = {}


def register_profile(profile: RadarDeviceProfile) -> None:
    """Register one supported profile."""
    if profile.name in _PROFILE_REGISTRY:
        raise ConfigurationError(f"Profile '{profile.name}' is already registered.")
    _PROFILE_REGISTRY[profile.name] = profile


register_profile(
    RadarDeviceProfile(
        name="generic-ti-dca1000",
        display_name="TI mmWave + DCA1000",
        radar_family="ti-mmwave",
        control_backend="ti-cli-uart",
        data_backend="dca1000-udp",
        supports_raw_capture=True,
        supports_live_view=True,
        supports_usb_telemetry=False,
        description="Generic TI mmWave CLI radar using DCA1000 raw LVDS capture.",
    )
)
register_profile(
    RadarDeviceProfile(
        name="iwr1843-dca1000",
        display_name="IWR1843 + DCA1000",
        radar_family="xwr18xx",
        control_backend="ti-cli-uart",
        data_backend="dca1000-udp",
        supports_raw_capture=True,
        supports_live_view=True,
        supports_usb_telemetry=False,
        description="IWR1843 with DCA1000 raw ADC capture over Ethernet.",
        default_cfg_path=Path("config/xwr18xx_profile_raw_capture.cfg"),
    )
)
register_profile(
    RadarDeviceProfile(
        name="iwr6843-dca1000",
        display_name="IWR6843 + DCA1000",
        radar_family="xwr68xx",
        control_backend="ti-cli-uart",
        data_backend="dca1000-udp",
        supports_raw_capture=True,
        supports_live_view=True,
        supports_usb_telemetry=False,
        description="IWR6843 with DCA1000 raw ADC capture over Ethernet.",
        default_cfg_path=Path("config/xwr68xx_profile_2026_04_02T12_19_57_603.cfg"),
    )
)
register_profile(
    RadarDeviceProfile(
        name="iwr6843aop-dca1000",
        display_name="IWR6843AOP + DCA1000",
        radar_family="xwr68xx",
        control_backend="ti-cli-uart",
        data_backend="dca1000-udp",
        supports_raw_capture=True,
        supports_live_view=True,
        supports_usb_telemetry=False,
        description="IWR6843AOP with DCA1000 raw ADC capture over Ethernet.",
        default_cfg_path=Path("config/iwr6843aop_profile_raw_capture.cfg"),
    )
)
register_profile(
    RadarDeviceProfile(
        name="iwr6843-usb",
        display_name="IWR6843 USB Telemetry",
        radar_family="xwr68xx",
        control_backend="ti-cli-uart",
        data_backend="ti-data-uart",
        supports_raw_capture=False,
        supports_live_view=True,
        supports_usb_telemetry=True,
        description="IWR6843 using the factory TI demo telemetry stream over USB UART.",
        default_cfg_path=Path("config/xwr68xx_profile_2026_04_02T12_19_57_603.cfg"),
    )
)
register_profile(
    RadarDeviceProfile(
        name="iwr6843aop-usb",
        display_name="IWR6843AOP USB Telemetry",
        radar_family="xwr68xx",
        control_backend="ti-cli-uart",
        data_backend="ti-data-uart",
        supports_raw_capture=False,
        supports_live_view=True,
        supports_usb_telemetry=True,
        description="IWR6843AOP using the factory TI demo telemetry stream over USB UART.",
        default_cfg_path=Path("config/iwr6843aop_usb_point_cloud.cfg"),
    )
)


def get_profile(name: str) -> RadarDeviceProfile:
    """Return one supported profile by name."""
    try:
        return _PROFILE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PROFILE_REGISTRY))
        raise ConfigurationError(
            f"Unknown profile '{name}'. Available profiles: {available}"
        ) from exc


def list_profiles() -> tuple[RadarDeviceProfile, ...]:
    """Return all supported profiles in a stable order."""
    return tuple(_PROFILE_REGISTRY[name] for name in sorted(_PROFILE_REGISTRY))


def list_profile_names() -> tuple[str, ...]:
    """Return all registered profile names."""
    return tuple(profile.name for profile in list_profiles())
