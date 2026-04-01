# mmwave_radar_devtool

Python toolkit for working with TI mmWave radar hardware, focused on the `IWR1843` radar and `DCA1000EVM` capture card.

## Features

- Configure the radar over the CLI serial port
- Probe the radar and capture-card setup before acquisition
- Record raw ADC data to disk
- Run a live terminal view during capture
- Generate capture-ready radar configuration files

## Installation

```bash
uv sync
```

For development dependencies:

```bash
uv sync --extra dev
```

## CLI

The package installs the `mmw` command.

### Probe hardware

```bash
uv run mmw probe \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg
```

### Capture to file

```bash
uv run mmw capture \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --output capture.bin \
  --duration 3
```

### Live terminal dashboard

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg
```

Optional bounded run:

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --duration 10
```

### Live terminal dashboard while recording

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --output capture.bin
```

### Plot a raw capture

```bash
uv run mmw plot \
  --input capture.bin \
  --samples 4096
```

### Generate a capture-ready config

```bash
uv run mmw make-capture-cfg \
  --input config/xwr18xx_profile_raw_capture.cfg \
  --output config/generated_capture.cfg
```

## Development

Run tests with:

```bash
uv run pytest
```

## License

This project is licensed under the Apache License 2.0. See `LICENSE`.
