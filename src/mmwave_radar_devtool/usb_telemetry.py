"""TI mmWave USB/UART telemetry parsing and streaming."""

from __future__ import annotations

import serial
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import CaptureConfig, RadarDataSerialConfig
from .exceptions import ConfigurationError, RadarSerialError

TI_MMWAVE_MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
TI_MMWAVE_HEADER_STRUCT = struct.Struct("<4H8I")
TI_MMWAVE_TLV_HEADER_STRUCT = struct.Struct("<II")
TI_MMWAVE_DETECTED_POINTS_TLV = 1
TI_MMWAVE_RANGE_PROFILE_TLV = 2
TI_MMWAVE_SIDE_INFO_TLV = 7


@dataclass(slots=True, frozen=True)
class TiRadarPoint:
    """One detected point from the TI demo telemetry stream."""

    x_m: float
    y_m: float
    z_m: float
    velocity_m_s: float
    snr_db: float | None = None
    noise_db: float | None = None

    @property
    def range_m(self) -> float:
        """Return radial distance from the sensor."""
        return (self.x_m**2 + self.y_m**2 + self.z_m**2) ** 0.5


@dataclass(slots=True, frozen=True)
class TiTelemetryTlv:
    """One TLV section from a TI telemetry packet."""

    tlv_type: int
    payload: bytes


@dataclass(slots=True, frozen=True)
class TiTelemetryFrame:
    """One decoded TI mmWave telemetry frame."""

    version: int
    total_packet_length: int
    platform: int
    frame_number: int
    time_cpu_cycles: int
    num_detected_objects: int
    num_tlvs: int
    subframe_number: int
    tlvs: tuple[TiTelemetryTlv, ...]
    points: tuple[TiRadarPoint, ...]
    range_profile: tuple[float, ...]

    @property
    def detected_points(self) -> int:
        """Return the number of parsed detected points."""
        return len(self.points)


@dataclass(slots=True, frozen=True)
class TelemetryStats:
    """Statistics from a USB telemetry session."""

    frames_received: int
    raw_bytes_read: int
    bytes_received: int
    tlvs_received: int
    detected_points_received: int
    elapsed_s: float
    first_frame_number: int | None
    last_frame_number: int | None
    frame_gaps_detected: int


class TiUsbTelemetryParser:
    """Incrementally parse TI mmWave UART telemetry frames."""

    def __init__(self) -> None:
        """Initialize the parser state."""
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[TiTelemetryFrame]:
        """Feed raw UART bytes and return any complete decoded frames."""
        if data:
            self._buffer.extend(data)

        frames: list[TiTelemetryFrame] = []
        while True:
            start = self._buffer.find(TI_MMWAVE_MAGIC_WORD)
            if start < 0:
                if len(self._buffer) > len(TI_MMWAVE_MAGIC_WORD):
                    self._buffer = self._buffer[-(len(TI_MMWAVE_MAGIC_WORD) - 1) :]
                break
            if start > 0:
                del self._buffer[:start]
            if len(self._buffer) < TI_MMWAVE_HEADER_STRUCT.size:
                break

            header = TI_MMWAVE_HEADER_STRUCT.unpack_from(self._buffer, 0)
            total_packet_length = int(header[5])
            if total_packet_length < TI_MMWAVE_HEADER_STRUCT.size:
                raise ConfigurationError(
                    f"Invalid TI telemetry packet length: {total_packet_length}"
                )
            if len(self._buffer) < total_packet_length:
                break

            packet = bytes(self._buffer[:total_packet_length])
            del self._buffer[:total_packet_length]
            frames.append(self._decode_packet(packet))
        return frames

    def _decode_packet(self, packet: bytes) -> TiTelemetryFrame:
        """Decode one complete TI telemetry packet."""
        (
            _magic0,
            _magic1,
            _magic2,
            _magic3,
            version,
            total_packet_length,
            platform,
            frame_number,
            time_cpu_cycles,
            num_detected_objects,
            num_tlvs,
            subframe_number,
        ) = TI_MMWAVE_HEADER_STRUCT.unpack_from(packet, 0)

        offset = TI_MMWAVE_HEADER_STRUCT.size
        tlvs: list[TiTelemetryTlv] = []
        points: tuple[TiRadarPoint, ...] = ()
        range_profile: tuple[float, ...] = ()
        pending_side_info: tuple[tuple[float, float], ...] = ()

        for _ in range(num_tlvs):
            if offset + TI_MMWAVE_TLV_HEADER_STRUCT.size > len(packet):
                break
            tlv_type, tlv_length = TI_MMWAVE_TLV_HEADER_STRUCT.unpack_from(packet, offset)
            if tlv_length < TI_MMWAVE_TLV_HEADER_STRUCT.size:
                break
            payload_start = offset + TI_MMWAVE_TLV_HEADER_STRUCT.size
            payload_end = offset + tlv_length
            if payload_end > len(packet):
                break
            payload = packet[payload_start:payload_end]
            tlvs.append(TiTelemetryTlv(tlv_type=tlv_type, payload=payload))
            if tlv_type == TI_MMWAVE_DETECTED_POINTS_TLV:
                points = self._parse_detected_points(payload)
            elif tlv_type == TI_MMWAVE_RANGE_PROFILE_TLV:
                range_profile = self._parse_range_profile(payload)
            elif tlv_type == TI_MMWAVE_SIDE_INFO_TLV:
                pending_side_info = self._parse_side_info(payload)
            offset = payload_end

        if points and pending_side_info:
            points = tuple(
                TiRadarPoint(
                    x_m=point.x_m,
                    y_m=point.y_m,
                    z_m=point.z_m,
                    velocity_m_s=point.velocity_m_s,
                    snr_db=pending_side_info[index][0] if index < len(pending_side_info) else None,
                    noise_db=pending_side_info[index][1]
                    if index < len(pending_side_info)
                    else None,
                )
                for index, point in enumerate(points)
            )

        return TiTelemetryFrame(
            version=version,
            total_packet_length=total_packet_length,
            platform=platform,
            frame_number=frame_number,
            time_cpu_cycles=time_cpu_cycles,
            num_detected_objects=num_detected_objects,
            num_tlvs=num_tlvs,
            subframe_number=subframe_number,
            tlvs=tuple(tlvs),
            points=points,
            range_profile=range_profile,
        )

    @staticmethod
    def _parse_detected_points(payload: bytes) -> tuple[TiRadarPoint, ...]:
        """Parse the standard floating-point detected-points TLV."""
        point_struct = struct.Struct("<ffff")
        points: list[TiRadarPoint] = []
        for offset in range(0, len(payload) - point_struct.size + 1, point_struct.size):
            x_m, y_m, z_m, velocity_m_s = point_struct.unpack_from(payload, offset)
            points.append(
                TiRadarPoint(
                    x_m=x_m,
                    y_m=y_m,
                    z_m=z_m,
                    velocity_m_s=velocity_m_s,
                )
            )
        return tuple(points)

    @staticmethod
    def _parse_side_info(payload: bytes) -> tuple[tuple[float, float], ...]:
        """Parse TI side-info TLV values as dB quantities."""
        side_info_struct = struct.Struct("<HH")
        values: list[tuple[float, float]] = []
        for offset in range(0, len(payload) - side_info_struct.size + 1, side_info_struct.size):
            snr, noise = side_info_struct.unpack_from(payload, offset)
            values.append((snr / 10.0, noise / 10.0))
        return tuple(values)

    @staticmethod
    def _parse_range_profile(payload: bytes) -> tuple[float, ...]:
        """Parse TI zero-Doppler range-profile bins as unsigned magnitudes."""
        if len(payload) < 2:
            return ()
        bin_count = len(payload) // 2
        return tuple(
            float(value) for value in struct.unpack(f"<{bin_count}H", payload[: bin_count * 2])
        )


@dataclass(slots=True)
class TelemetryMetrics:
    """Mutable telemetry statistics used by the USB live dashboard."""

    started_at: float = field(default_factory=time.monotonic)
    frames_received: int = 0
    bytes_received: int = 0
    tlvs_received: int = 0
    detected_points_received: int = 0
    first_frame_number: int | None = None
    last_frame_number: int | None = None
    frame_gaps_detected: int = 0

    def record_frame(self, frame: TiTelemetryFrame) -> None:
        """Update metrics from one decoded frame."""
        self.frames_received += 1
        self.bytes_received += frame.total_packet_length
        self.tlvs_received += len(frame.tlvs)
        self.detected_points_received += frame.detected_points
        if self.first_frame_number is None:
            self.first_frame_number = frame.frame_number
        elif (
            self.last_frame_number is not None and frame.frame_number != self.last_frame_number + 1
        ):
            self.frame_gaps_detected += max(0, frame.frame_number - self.last_frame_number - 1)
        self.last_frame_number = frame.frame_number


class TiUsbTelemetryStream:
    """Read TI mmWave telemetry frames from the USB/UART data port."""

    def __init__(self, config: RadarDataSerialConfig) -> None:
        """Initialize the telemetry stream reader."""
        self._config = config
        self._serial: serial.Serial | None = None

    def open(self) -> None:
        """Open the telemetry serial port."""
        if self._serial is not None:
            return
        self._serial = serial.Serial(
            port=self._config.data_port,
            baudrate=self._config.data_baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self._config.read_timeout_s,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        self._serial.reset_input_buffer()

    def close(self) -> None:
        """Close the telemetry serial port."""
        if self._serial is None:
            return
        self._serial.close()
        self._serial = None

    def __enter__(self) -> TiUsbTelemetryStream:
        """Open the telemetry reader context."""
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close the telemetry reader context."""
        self.close()

    def probe(self) -> str:
        """Verify that the telemetry serial port can be opened."""
        self.open()
        return "opened"

    def stream(
        self,
        capture_config: CaptureConfig,
        *,
        frame_consumer: Callable[[TiTelemetryFrame], None] | None = None,
        stop_condition: Callable[[], bool] | None = None,
    ) -> TelemetryStats:
        """Stream TI telemetry frames for a bounded or continuous session."""
        if self._serial is None:
            raise RadarSerialError("Radar data serial port is not open")

        parser = TiUsbTelemetryParser()
        output_path = (
            Path(capture_config.output_path) if capture_config.output_path is not None else None
        )
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("wb") if output_path is not None else None

        metrics = TelemetryMetrics()
        start = time.monotonic()
        first_frame_deadline = start + self._config.first_frame_timeout_s
        raw_bytes_read = 0
        try:
            while True:
                if (
                    capture_config.duration_s is not None
                    and time.monotonic() - start >= capture_config.duration_s
                ):
                    break
                if stop_condition is not None and stop_condition():
                    break

                waiting = self._serial.in_waiting
                try:
                    chunk = self._serial.read(waiting or 4096)
                except KeyboardInterrupt:
                    break
                if not chunk:
                    if metrics.frames_received == 0 and time.monotonic() >= first_frame_deadline:
                        raise ConfigurationError(self._build_no_frame_error(raw_bytes_read))
                    time.sleep(self._config.poll_interval_s)
                    continue
                raw_bytes_read += len(chunk)
                if output_handle is not None:
                    output_handle.write(chunk)
                for frame in parser.feed(chunk):
                    metrics.record_frame(frame)
                    if frame_consumer is not None:
                        frame_consumer(frame)
                if metrics.frames_received == 0 and time.monotonic() >= first_frame_deadline:
                    raise ConfigurationError(self._build_no_frame_error(raw_bytes_read))
        finally:
            if output_handle is not None:
                output_handle.close()

        return TelemetryStats(
            frames_received=metrics.frames_received,
            raw_bytes_read=raw_bytes_read,
            bytes_received=metrics.bytes_received,
            tlvs_received=metrics.tlvs_received,
            detected_points_received=metrics.detected_points_received,
            elapsed_s=time.monotonic() - start,
            first_frame_number=metrics.first_frame_number,
            last_frame_number=metrics.last_frame_number,
            frame_gaps_detected=metrics.frame_gaps_detected,
        )

    @staticmethod
    def _build_no_frame_error(raw_bytes_read: int) -> str:
        """Build a user-facing error when no telemetry frames were decoded."""
        if raw_bytes_read == 0:
            return (
                "No bytes were received from the radar data UART. "
                "Check that --radar-data-port points to the telemetry port, the sensor is streaming, and the USB cable is connected."
            )
        return (
            "Received bytes from the radar data UART but could not decode any TI telemetry frames. "
            "This usually means the wrong baud rate was used, the wrong serial port was selected, or the firmware output format does not match the expected TI demo telemetry stream."
        )
