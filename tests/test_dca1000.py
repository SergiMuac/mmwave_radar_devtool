"""Tests for DCA1000 packet helpers."""

from mmwave_radar_devtool.config import DCA1000Config, DCA1000DataLoggingMode
from mmwave_radar_devtool.dca1000 import DCA1000Client, DCA1000DataPacket


def test_configure_fpga_payload_matches_ti_observed_defaults() -> None:
    """The FPGA config payload should match the known 18xx raw capture pattern."""
    client = DCA1000Client(DCA1000Config())

    payload = client._UINT8_SEXTUPLE.pack(
        int(DCA1000DataLoggingMode.RAW),
        int(client.config.device_mode),
        int(client.config.capture_interface),
        int(client.config.stream_transport),
        3,
        client.config.fpga_config_timer_s,
    )

    assert payload.hex() == "01020102031e"


def test_configure_packet_payload_matches_ti_observed_defaults() -> None:
    """The packet config payload should match the known 18xx raw capture pattern."""
    client = DCA1000Client(DCA1000Config(packet_delay_us=10, packet_payload_size_bytes=1456))

    ethernet_packet_size = client.config.packet_payload_size_bytes + 14
    packet_delay_ticks = int(round(client.config.packet_delay_us * 312.5))
    payload = client._UINT16_TRIPLE.pack(ethernet_packet_size, packet_delay_ticks, 0)

    assert payload.hex() == "be05350c0000"


def test_data_packet_parser_extracts_header_and_payload() -> None:
    """The UDP data parser should decode sequence and byte count."""
    datagram = (
        (7).to_bytes(4, byteorder="little") + (1234).to_bytes(6, byteorder="little") + b"abcdef"
    )

    packet = DCA1000DataPacket.from_udp_datagram(datagram)

    assert packet.sequence_number == 7
    assert packet.byte_count == 1234
    assert packet.payload == b"abcdef"
