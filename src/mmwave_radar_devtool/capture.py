"""Capture orchestration and UDP data sinks."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .cfg_parser import RadarCliConfig
from .config import CaptureConfig, DCA1000Config, RadarDataSerialConfig, RadarSerialConfig
from .dca1000 import DCA1000Client, DCA1000DataPacket, map_capture_requirements
from .exceptions import ConfigurationError
from .live_view import TerminalLiveDashboard
from .profiles import RadarDeviceProfile
from .serial_control import RadarSerialController
from .telemetry_live import TerminalTelemetryDashboard
from .usb_telemetry import TelemetryStats, TiTelemetryFrame, TiUsbTelemetryStream


class PacketConsumer(Protocol):
    """Callback protocol for live packet handling."""

    def __call__(self, packet: DCA1000DataPacket) -> None:
        """Handle one decoded UDP packet."""


@dataclass(slots=True, frozen=True)
class CaptureStats:
    """Statistics from a capture run."""

    packets_received: int
    bytes_received: int
    payload_bytes_written: int
    elapsed_s: float
    first_sequence_number: int | None
    last_sequence_number: int | None
    sequence_gaps_detected: int


class UdpCaptureSink:
    """Receive DCA1000 data packets for file capture or live viewing."""

    def __init__(self, config: DCA1000Config) -> None:
        """Initialize the sink."""
        self._config = config

    def capture_stream(
        self,
        capture_config: CaptureConfig,
        *,
        packet_consumer: PacketConsumer | None = None,
        stop_condition: Callable[[], bool] | None = None,
    ) -> CaptureStats:
        """Capture UDP payloads for a bounded or continuous session and optionally save them."""
        output_path = (
            Path(capture_config.output_path) if capture_config.output_path is not None else None
        )
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, self._config.data_socket_buffer_bytes
        )
        udp_socket.bind((str(self._config.host_ip), self._config.data_port))
        udp_socket.settimeout(capture_config.socket_timeout_s)

        start = time.monotonic()
        packets_received = 0
        bytes_received = 0
        payload_bytes_written = 0
        first_sequence_number: int | None = None
        last_sequence_number: int | None = None
        sequence_gaps_detected = 0
        file_handle = output_path.open("wb") if output_path is not None else None

        try:
            while True:
                if (
                    capture_config.duration_s is not None
                    and time.monotonic() - start >= capture_config.duration_s
                ):
                    break
                if stop_condition is not None and stop_condition():
                    break
                try:
                    datagram, _ = udp_socket.recvfrom(capture_config.packet_size_bytes)
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    break

                packets_received += 1
                bytes_received += len(datagram)
                data_packet = DCA1000DataPacket.from_udp_datagram(datagram)

                if first_sequence_number is None:
                    first_sequence_number = data_packet.sequence_number
                elif (
                    last_sequence_number is not None
                    and data_packet.sequence_number != last_sequence_number + 1
                ):
                    sequence_gaps_detected += max(
                        0, data_packet.sequence_number - last_sequence_number - 1
                    )

                last_sequence_number = data_packet.sequence_number

                if file_handle is not None:
                    if capture_config.strip_dca_header:
                        file_handle.write(data_packet.payload)
                        payload_bytes_written += len(data_packet.payload)
                    else:
                        file_handle.write(datagram)
                        payload_bytes_written += len(datagram)

                if packet_consumer is not None:
                    packet_consumer(data_packet)
        finally:
            if file_handle is not None:
                file_handle.close()
            udp_socket.close()

        elapsed_s = time.monotonic() - start
        return CaptureStats(
            packets_received=packets_received,
            bytes_received=bytes_received,
            payload_bytes_written=payload_bytes_written,
            elapsed_s=elapsed_s,
            first_sequence_number=first_sequence_number,
            last_sequence_number=last_sequence_number,
            sequence_gaps_detected=sequence_gaps_detected,
        )

    def capture_to_file(self, capture_config: CaptureConfig) -> CaptureStats:
        """Capture raw UDP payloads to disk for a fixed duration."""
        return self.capture_stream(capture_config)


class CaptureOrchestrator:
    """High-level orchestration for configuration and raw capture."""

    def __init__(
        self,
        dca_config: DCA1000Config,
        serial_config: RadarSerialConfig,
        profile: RadarDeviceProfile,
        data_serial_config: RadarDataSerialConfig | None = None,
    ) -> None:
        """Initialize the orchestrator."""
        self._dca_config = dca_config
        self._serial_config = serial_config
        self._profile = profile
        self._data_serial_config = data_serial_config

    def probe(self, cfg: RadarCliConfig | None = None) -> dict[str, str]:
        """Verify communication with DCA1000 and optionally the radar CLI."""
        results: dict[str, str] = {}
        if self._profile.data_backend == "dca1000-udp":
            with DCA1000Client(self._dca_config) as dca:
                dca.system_connect()
                results["dca1000"] = "connected"
        elif self._profile.data_backend == "ti-data-uart":
            with TiUsbTelemetryStream(self._require_data_serial_config()) as telemetry:
                results["radar_data_uart"] = telemetry.probe()

        if cfg is not None:
            with RadarSerialController(self._serial_config) as radar:
                probe_command = cfg.commands[0].text if cfg.commands else "version"
                response = radar.send_command(probe_command)
                results["radar_cli"] = response.strip() or "command sent"
                results["radar_cli_probe_command"] = probe_command
        return results

    def capture(self, cfg: RadarCliConfig, capture_config: CaptureConfig) -> CaptureStats:
        """Apply radar config, configure DCA, capture UDP packets, then stop cleanly."""
        if not self._profile.supports_raw_capture:
            raise ConfigurationError(
                f"Profile '{self._profile.name}' does not support raw capture. Use 'live' with USB telemetry instead."
            )
        requirements = cfg.extract_capture_requirements()
        dca_settings = map_capture_requirements(requirements)
        cfg_commands_before_sensor_start = cfg.command_texts_excluding(("sensorStart",))
        sink = UdpCaptureSink(self._dca_config)

        with (
            DCA1000Client(self._dca_config) as dca,
            RadarSerialController(self._serial_config) as radar,
        ):
            radar.send_cfg_lines(cfg_commands_before_sensor_start)
            dca.configure_for_recording(
                data_logging_mode=dca_settings.data_logging_mode,
                data_format_mode=dca_settings.data_format_mode,
            )
            dca.start_record()
            radar.sensor_start()
            try:
                stats = sink.capture_stream(capture_config)
            finally:
                dca.stop_record()
                radar.sensor_stop()
        return stats

    def capture_live(
        self, cfg: RadarCliConfig, capture_config: CaptureConfig
    ) -> CaptureStats | TelemetryStats:
        """Run a live terminal dashboard for the active transport."""
        if self._profile.data_backend == "ti-data-uart":
            return self.capture_live_telemetry(cfg=cfg, capture_config=capture_config)

        requirements = cfg.extract_capture_requirements()
        dca_settings = map_capture_requirements(requirements)
        cfg_commands_before_sensor_start = cfg.command_texts_excluding(("sensorStart",))
        sink = UdpCaptureSink(self._dca_config)
        dashboard = TerminalLiveDashboard(radar_cfg=cfg, title=f"{self._profile.display_name} Live")

        with (
            DCA1000Client(self._dca_config) as dca,
            RadarSerialController(self._serial_config) as radar,
        ):
            radar.send_cfg_lines(cfg_commands_before_sensor_start)
            dca.configure_for_recording(
                data_logging_mode=dca_settings.data_logging_mode,
                data_format_mode=dca_settings.data_format_mode,
            )
            dashboard.start()
            try:
                dca.start_record()
                radar.sensor_start()
                stats = sink.capture_stream(
                    capture_config,
                    packet_consumer=self._build_live_consumer(dashboard),
                    stop_condition=lambda: dashboard.stop_requested,
                )
            finally:
                try:
                    dca.stop_record()
                finally:
                    try:
                        radar.sensor_stop()
                    finally:
                        dashboard.stop()
        return stats

    def capture_live_telemetry(
        self, cfg: RadarCliConfig, capture_config: CaptureConfig
    ) -> TelemetryStats:
        """Run a live dashboard while reading TI telemetry over the data UART."""
        cfg_commands_before_sensor_start = cfg.command_texts_excluding(("sensorStart",))
        dashboard = TerminalTelemetryDashboard(
            title=f"{self._profile.display_name} Live",
            radar_cfg=cfg,
        )
        telemetry = TiUsbTelemetryStream(self._require_data_serial_config())

        with telemetry, RadarSerialController(self._serial_config) as radar:
            radar.send_cfg_lines(cfg_commands_before_sensor_start)
            dashboard.start()
            try:
                radar.sensor_start()
                stats = telemetry.stream(
                    capture_config,
                    frame_consumer=self._build_telemetry_live_consumer(dashboard),
                    stop_condition=lambda: dashboard.stop_requested,
                )
            finally:
                try:
                    radar.sensor_stop()
                finally:
                    dashboard.stop()
        return stats

    @staticmethod
    def _build_live_consumer(dashboard: TerminalLiveDashboard) -> PacketConsumer:
        """Create a packet consumer that refreshes the terminal dashboard."""
        last_render_at = time.monotonic()

        def consume(packet: DCA1000DataPacket) -> None:
            nonlocal last_render_at
            dashboard.metrics.record_packet(packet)
            now = time.monotonic()
            if now - last_render_at >= 0.1:
                dashboard.update()
                last_render_at = now

        return consume

    @staticmethod
    def _build_telemetry_live_consumer(
        dashboard: TerminalTelemetryDashboard,
    ) -> Callable[[TiTelemetryFrame], None]:
        """Create a frame consumer that refreshes the telemetry dashboard."""
        last_render_at = time.monotonic()

        def consume(frame: TiTelemetryFrame) -> None:
            nonlocal last_render_at
            dashboard.metrics.record_frame(frame)
            now = time.monotonic()
            if now - last_render_at >= 0.1:
                dashboard.update()
                last_render_at = now

        return consume

    def _require_data_serial_config(self) -> RadarDataSerialConfig:
        """Return the configured data UART settings for USB telemetry profiles."""
        if self._data_serial_config is None:
            raise ConfigurationError(
                f"Profile '{self._profile.name}' requires --radar-data-port for USB telemetry."
            )
        return self._data_serial_config
