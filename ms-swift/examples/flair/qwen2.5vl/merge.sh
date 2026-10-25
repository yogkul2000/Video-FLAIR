#!/usr/bin/env bash
set -euo pipefail
: "${ADAPTER:?set ADAPTER to a LoRA checkpoint dir}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
swift export --adapters "$ADAPTER" --merge_lora true --output_dir "$OUTPUT_DIR" "$@"
