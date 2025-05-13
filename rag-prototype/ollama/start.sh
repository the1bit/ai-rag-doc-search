#!/bin/bash

set -euo pipefail

MODEL_NAME="mistral"

echo "[INFO] Starting Ollama in background..."
ollama serve &
OLLAMA_PID=$!

echo "[INFO] Waiting for Ollama API to be ready..."
until curl -s -o /dev/null -w "%{http_code}" http://localhost:11434 | grep -q "200"; do
  sleep 1
done

echo "[INFO] Checking if model '$MODEL_NAME' is available..."
if ! ollama list | grep -q "$MODEL_NAME"; then
  echo "[INFO] Model '$MODEL_NAME' not found. Pulling..."
  ollama pull "$MODEL_NAME"
else
  echo "[INFO] Model '$MODEL_NAME' already present."
fi

echo "[INFO] Ollama is ready and model '$MODEL_NAME' is loaded."

# Keep container alive
wait $OLLAMA_PID
