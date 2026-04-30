#!/usr/bin/env bash
# Download the Piper de_DE-thorsten-high voice model.
# Phase 2 — this script is a placeholder until the TTS service is wired up.

set -euo pipefail

MODEL_DIR="$(dirname "$0")/../data/piper_models"
MODEL_FILE="de_DE-thorsten-high.onnx"
MODEL_JSON="de_DE-thorsten-high.onnx.json"

mkdir -p "$MODEL_DIR"

BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high"

echo "[download_piper_model] Phase 2 — fetching Piper voice model to $MODEL_DIR"
curl -L -o "$MODEL_DIR/$MODEL_FILE"  "$BASE_URL/$MODEL_FILE"
curl -L -o "$MODEL_DIR/$MODEL_JSON" "$BASE_URL/$MODEL_JSON"

echo "[download_piper_model] Done."
