"""UDP client for the DCA1000EVM control and data channels."""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from enum import IntEnum

from .cfg_parser import RadarCaptureRequirements
from .config import DCA1000Config, DCA1000DataLoggingMode
from .exceptions import DCA1000ResponseError


class DCA1000Command(IntEnum):
    """Known DCA1000 command codes."""

    RESET_FPGA = 0x01
    RESET_AR_DEV = 0x02
    CONFIG_FPGA_GEN = 0x03
    CONFIG_EEPROM = 0x04
    RECORD_START = 0x05
    RECORD_STOP = 0x06
    PLAYBACK_START = 0x07
    PLAYBACK_STOP = 0x08
    SYSTEM_CONNECT = 0x09
    SYSTEM_ERROR = 0x0A
    CONFIG_PACKET_DATA = 0x0B
    CONFIG_DATA_MODE_AR_DEV = 0x0C
    INIT_FPGA_PLAYBACK = 0x0D
    READ_FPGA_VERSION = 0x0E


@dataclass(slots=True, frozen=True)
class DCA1000Response:
    """Decoded DCA1000 response packet."""

    command: DCA1000Command
    status: int


@dataclass(slots=True, frozen=True)
class DCA1000DataPacket:
    """Decoded UDP packet emitted by the DCA1000 data port."""

    sequence_number: int
    byte_count: int
    payload: bytes

    _HEADER_SIZE = 10

    @classmethod
    def from_udp_datagram(cls, datagram: bytes) -> DCA1000DataPacket:
        """Parse one DCA1000 UDP data datagram."""
        if len(datagram) < cls._HEADER_SIZE:
            raise DCA1000ResponseError(f"DCA1000 data packet is too short: {len(datagram)} bytes")

        sequence_number = struct.unpack_from("<I", datagram, 0)[0]
        byte_count = int.from_bytes(datagram[4:10], byteorder="little", signed=False)
        payload = datagram[10:]
        return cls(sequence_number=sequence_number, byte_count=byte_count, payload=payload)


@dataclass(slots=True, frozen=True)
class DCA1000CaptureSettings:
    """DCA1000-specific settings derived from generic capture requirements."""

    data_logging_mode: DCA1000DataLoggingMode
    data_format_mode: int


def map_capture_requirements(requirements: RadarCaptureRequirements) -> DCA1000CaptureSettings:
    """Translate generic TI raw-capture requirements into DCA1000 settings."""
    data_logging_mode = (
        DCA1000DataLoggingMode.MULTI
        if requirements.lvds_stream_cfg.enable_header
        else DCA1000DataLoggingMode.RAW
    )
    return DCA1000CaptureSettings(
        data_logging_mode=data_logging_mode,
        data_format_mode=requirements.adc_cfg.data_format_mode,
    )


class DCA1000Client:
    """Client for sending control commands to the DCA1000EVM."""

    _HEADER = 0xA55A
    _FOOTER = 0xEEAA
    _REQUEST_STRUCT = struct.Struct("<HHH")
    _RESPONSE_STRUCT = struct.Struct("<HHH")
    _UINT16_TRIPLE = struct.Struct("<HHH")
    _UINT8_SEXTUPLE = struct.Struct("<6B")

    def __init__(self, config: DCA1000Config) -> None:
        """Initialize the client."""
        self._config = config
        self._socket: socket.socket | None = None

    @property
    def config(self) -> DCA1000Config:
        """Return the active configuration."""
        return self._config

    def open(self) -> None:
        """Open and bind the UDP socket."""
        if self._socket is not None:
            return
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.settimeout(self._config.command_timeout_s)
        udp_socket.bind((str(self._config.host_ip), self._config.config_port))
        self._socket = udp_socket

    def close(self) -> None:
        """Close the UDP socket."""
        if self._socket is None:
            return
        self._socket.close()
        self._socket = None

    def __enter__(self) -> DCA1000Client:
        """Open the client context."""
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close the client context."""
        self.close()

    def send_command(self, command: DCA1000Command, payload: bytes = b"") -> DCA1000Response:
        """Send a command and decode the response."""
        if self._socket is None:
            raise DCA1000ResponseError("DCA1000 control socket is not open")

        packet = self._build_request(command=command, payload=payload)
        self._socket.sendto(packet, (str(self._config.fpga_ip), self._config.config_port))
        response, _ = self._socket.recvfrom(4096)
        return self._parse_response(expected_command=command, response=response)

    def system_connect(self) -> DCA1000Response:
        """Send SYSTEM_CONNECT."""
        return self.send_command(DCA1000Command.SYSTEM_CONNECT)

    def reset_fpga(self) -> DCA1000Response:
        """Send RESET_FPGA."""
        return self.send_command(DCA1000Command.RESET_FPGA)

    def configure_fpga(
        self,
        *,
        data_logging_mode: DCA1000DataLoggingMode,
        data_format_mode: int,
    ) -> DCA1000Response:
        """Send CONFIG_FPGA_GEN using the TI LVDS-over-Ethernet defaults."""
        payload = self._UINT8_SEXTUPLE.pack(
            int(data_logging_mode),
            int(self._config.device_mode),
            int(self._config.capture_interface),
            int(self._config.stream_transport),
            data_format_mode,
            self._config.fpga_config_timer_s,
        )
        return self.send_command(DCA1000Command.CONFIG_FPGA_GEN, payload=payload)

    def configure_packet_data(self) -> DCA1000Response:
        """Send CONFIG_PACKET_DATA using the standard DCA1000 packet settings."""
        ethernet_packet_size = self._config.packet_payload_size_bytes + 14
        packet_delay_ticks = int(round(self._config.packet_delay_us * 312.5))
        payload = self._UINT16_TRIPLE.pack(ethernet_packet_size, packet_delay_ticks, 0)
        return self.send_command(DCA1000Command.CONFIG_PACKET_DATA, payload=payload)

    def configure_for_recording(
        self,
        *,
        data_logging_mode: DCA1000DataLoggingMode,
        data_format_mode: int,
    ) -> tuple[DCA1000Response, DCA1000Response, DCA1000Response]:
        """Send the DCA1000 commands required before RECORD_START."""
        response_connect = self.system_connect()
        response_fpga = self.configure_fpga(
            data_logging_mode=data_logging_mode,
            data_format_mode=data_format_mode,
        )
        response_packet = self.configure_packet_data()
        return response_connect, response_fpga, response_packet

    def start_record(self) -> DCA1000Response:
        """Send RECORD_START."""
        return self.send_command(DCA1000Command.RECORD_START)

    def stop_record(self) -> DCA1000Response:
        """Send RECORD_STOP."""
        return self.send_command(DCA1000Command.RECORD_STOP)

    def read_fpga_version(self) -> DCA1000Response:
        """Send READ_FPGA_VERSION."""
        return self.send_command(DCA1000Command.READ_FPGA_VERSION)

    @classmethod
    def _build_request(cls, command: DCA1000Command, payload: bytes) -> bytes:
        """Build a DCA1000 command request packet."""
        header = cls._REQUEST_STRUCT.pack(cls._HEADER, int(command), len(payload))
        footer = struct.pack("<H", cls._FOOTER)
        return header + payload + footer

    @classmethod
    def _parse_response(
        cls,
        *,
        expected_command: DCA1000Command,
        response: bytes,
    ) -> DCA1000Response:
        """Decode a DCA1000 command response packet."""
        minimum_size = cls._RESPONSE_STRUCT.size + 2
        if len(response) < minimum_size:
            raise DCA1000ResponseError(f"Response too short: {len(response)} bytes")

        header, command_value, status = cls._RESPONSE_STRUCT.unpack_from(response, 0)
        footer = struct.unpack_from("<H", response, cls._RESPONSE_STRUCT.size)[0]

        if header != cls._HEADER:
            raise DCA1000ResponseError(f"Unexpected response header: 0x{header:04X}")
        if footer != cls._FOOTER:
            raise DCA1000ResponseError(f"Unexpected response footer: 0x{footer:04X}")

        command = DCA1000Command(command_value)
        if command is not expected_command:
            raise DCA1000ResponseError(
                f"Expected response for {expected_command.name}, got {command.name}"
            )
        if status != 0:
            raise DCA1000ResponseError(f"DCA1000 returned failure for {command.name}")
        return DCA1000Response(command=command, status=status)
