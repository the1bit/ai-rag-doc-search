# ai-rag-doc-search
Document search with RAG


# API

How to preapre api for local running

```bash
cd rag-prototype/api

poetry install 3.11

poetry env activate
```


podman compose down -v
docker-compose down -v   


podman compose up --build 


docker-compose build ollama
docker-compose up


curl -X POST http://localhost:8900/init-question \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "session_id": "test1",
    "question": "What is the name of the TFLint configuration file?",
}'


curl -N -H "Content-Type: application/json" \
  -d '{"session_id":"tflint01", "question":"What is the name of the TFLint configuration file?"}' \
  http://localhost:8900/init-question



curl -N -H "Content-Type: application/json" \
  -d '{"session_id":"tflint01", "question":"Is there any plugin for TFLint?"}' \
  http://localhost:8900/conversation



curl -X POST http://localhost:8900/new-conversation \
    -H "Content-Type: application/json" \
    -d '{"session_id": "user123"}'

curl -X POST http://localhost:8900/reset-vector-store

curl -X POST http://localhost:8900/load-documents