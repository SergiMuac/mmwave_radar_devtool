#!/usr/bin/env bash

set -u

prompt_label() {
  local input

  while true; do
    read -r -p "Enter label for this recording: " input
    if [[ -n "$input" ]]; then
      printf '%s\n' "$input"
      return
    fi

    echo "Label cannot be empty."
  done
}

label="$(prompt_label)"
record_num=0

while true; do
  output="${label}_${record_num}.bin"

  echo "Recording to: ${output}"
  uv run mmw capture \
    --cfg config/xwr18xx_profile_raw_capture.cfg \
    --radar-cli-port /dev/ttyACM0 \
    --duration 5 \
    --output "$output"

  while true; do
    read -r -p "Next action: [Enter/c] continue, [l] change label, [q] quit: " action

    case "$action" in
      ""|c|C)
        record_num=$((record_num + 1))
        break
        ;;
      l|L)
        label="$(prompt_label)"
        record_num=0
        break
        ;;
      q|Q)
        echo "Done."
        exit 0
        ;;
      *)
        echo "Please enter c, l, or q."
        ;;
    esac
  done
done
