"""Data models for runtime configuration."""

from __future__ import annotations

from enum import IntEnum
from ipaddress import IPv4Address
from pathlib import Path

from pydantic import BaseModel, Field, PositiveInt


class DCA1000DataLoggingMode(IntEnum):
    """DCA1000 data logging modes."""

    RAW = 1
    MULTI = 2


class DCA1000DeviceMode(IntEnum):
    """Observed device selector values for DCA1000 FPGA configuration."""

    XWR16XX_XWR18XX_XWR68XX = 2


class DCA1000CaptureInterface(IntEnum):
    """Capture interface selector."""

    LVDS = 1


class DCA1000StreamTransport(IntEnum):
    """Streaming transport selector."""

    ETHERNET = 2


class DCA1000Config(BaseModel):
    """Configuration for DCA1000 UDP communication."""

    fpga_ip: IPv4Address = IPv4Address("192.168.33.180")
    host_ip: IPv4Address = IPv4Address("192.168.33.30")
    config_port: PositiveInt = 4096
    data_port: PositiveInt = 4098
    command_timeout_s: float = Field(default=2.0, gt=0.0)
    data_socket_buffer_bytes: int = Field(default=4 * 1024 * 1024, ge=65536)
    fpga_config_timer_s: int = Field(default=30, ge=1, le=255)
    packet_delay_us: int = Field(default=10, ge=0, le=255)
    packet_payload_size_bytes: int = Field(default=1456, ge=64, le=1456)
    lvds_mode: int = Field(default=2, ge=1, le=4)
    device_mode: DCA1000DeviceMode = DCA1000DeviceMode.XWR16XX_XWR18XX_XWR68XX
    capture_interface: DCA1000CaptureInterface = DCA1000CaptureInterface.LVDS
    stream_transport: DCA1000StreamTransport = DCA1000StreamTransport.ETHERNET


class RadarSerialConfig(BaseModel):
    """Configuration for radar CLI UART."""

    cli_port: str
    cli_baudrate: PositiveInt = 115200
    cli_timeout_s: float = Field(default=1.0, gt=0.0)
    command_timeout_s: float = Field(default=2.0, gt=0.0)
    startup_drain_timeout_s: float = Field(default=0.5, gt=0.0)
    prompt_idle_timeout_s: float = Field(default=0.2, gt=0.0)
    post_config_settle_s: float = Field(default=0.25, gt=0.0)
    poll_interval_s: float = Field(default=0.002, gt=0.0)
    debug_serial: bool = False
    verbose: bool = False


class RadarDataSerialConfig(BaseModel):
    """Configuration for radar USB/UART telemetry output."""

    data_port: str
    data_baudrate: PositiveInt = 921600
    read_timeout_s: float = Field(default=0.1, gt=0.0)
    poll_interval_s: float = Field(default=0.01, gt=0.0)
    first_frame_timeout_s: float = Field(default=5.0, gt=0.0)


class CaptureConfig(BaseModel):
    """Configuration for a raw UDP capture session."""

    output_path: Path | None = None
    duration_s: float | None = Field(default=None)
    packet_size_bytes: int = Field(default=4096, ge=64)
    strip_dca_header: bool = True
    socket_timeout_s: float = Field(default=0.1, gt=0.0)
