#!/bin/bash

MODEL_NAME="mistral"

echo "[INFO] Starting Ollama in background..."
ollama serve &
OLLAMA_PID=$!

echo "[INFO] Waiting for Ollama API to be ready..."
until curl -s http://localhost:11434 > /dev/null; do
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

# Wait for background Ollama process to keep container alive
wait $OLLAMA_PID
