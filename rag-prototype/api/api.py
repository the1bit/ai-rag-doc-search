from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import redis.asyncio as redis
import os
import json
import logging
from pathlib import Path
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # vagy DEBUG, ha részletesebb logolás kell
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Redis utils
async def get_conversation(session_id: str) -> List[Dict[str, str]]:
    data = await redis_client.get(session_id)
    if data:
        return json.loads(data)
    return []

async def save_conversation(session_id: str, history: List[Dict[str, str]]):
    await redis_client.set(session_id, json.dumps(history))


async def stream_answer(chain: Runnable, variables: dict):
    async for chunk in chain.astream(variables):
        yield chunk.content

# Async generator to stream and collect the full response
async def stream_and_store_answer(chain: Runnable, variables: dict, session_id: str, question: str):
    full_answer = ""

    async for chunk in chain.astream(variables):
        content = chunk  
        full_answer += content
        yield content

    # Save to Redis after the full response is streamed
    try:
        history = [{"question": question, "answer": full_answer}]
        await save_conversation(session_id, history)
    except Exception as e:
        logger.error(f"Failed to save conversation for session {session_id}: {e}")


# Async generator to stream and save conversation response
async def stream_and_store_conversation_answer(
    chain: Runnable,
    variables: dict,
    session_id: str,
    question: str
):
    full_answer = ""
    async for chunk in chain.astream(variables):
        content = chunk.content
        full_answer += content
        yield content

    try:
        history = await get_conversation(session_id)
        history.append({"question": question, "answer": full_answer})
        await save_conversation(session_id, history)
    except Exception as e:
        logger.error(f"Failed to save conversation for session {session_id}: {e}")


async def stream_json_response(chain: Runnable, variables: dict, session_id: str, question: str):
    full_answer = ""
    yield '{"response":"'

    try:
        async for chunk in chain.astream(variables):
            content = chunk.replace('"', '\\"').replace("\n", "\\n")
            full_answer += content
            logger.debug(f"Streaming chunk: {repr(content)}")
            yield content
    except Exception as e:
        logger.error(f"Streaming error: {e}")
    finally:
        yield '"}'  # mindig zárjuk le a JSON választ

        try:
            history = await get_conversation(session_id)
            history.append({"question": question, "answer": full_answer})
            await save_conversation(session_id, history)
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")


# Initialize FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # vagy ["*"] fejlesztéshez
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
INIT_PROMPT_TEMPLATE = os.getenv("INIT_PROMPT_TEMPLATE", "./prompts/init_prompt_en.tmpl")
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE", "./prompts/prompt_en.tmpl")

# Initialize Redis client
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)

# Initialize LLM and embedding model
llm = ChatOllama(
    model="mistral", 
    base_url=OLLAMA_BASE_URL,
    streaming=True
    )
embedding = OllamaEmbeddings(model="mistral", base_url=OLLAMA_BASE_URL)

# Prompt template
base_prompt = ChatPromptTemplate.from_template(
    "Answer the following question using the provided context.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}"
)


# Create Qdrant collection if it doesn't exist
def create_qdrant_collection():
    # Connect to Qdrant
    logger.info("Connecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    logger.info("Checking if collection exists...")
    # Create collection if it doesn't exist
    if not client.collection_exists("texts"):
        logger.info("Creating collection...")
        client.create_collection(
            collection_name="texts",
            vectors_config=VectorParams(
                size=4096,  # Size of the embedding vector
                distance=Distance.COSINE,
            ),
        )

# Delete conversation from Redis
async def delete_conversation(session_id: str):
    await redis_client.delete(session_id)


output_parser = StrOutputParser()

# Question input model
class QuestionInput(BaseModel):
    session_id: str
    question: str

# Conversation input model
class ConversationInput(BaseModel):
    session_id: str
    question: str

# Input model for conversation history
class SessionInput(BaseModel):
    session_id: str


################################
# API Endpoints
################################



# Initialize question endpoint with streaming
@app.post("/init-question")
async def init_question(input: QuestionInput):
    session_id = input.session_id
    question = input.question
    logger.debug(f"Received question: {question} for session: {session_id}")

    create_qdrant_collection()

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vectorstore = Qdrant(
        client=client,
        collection_name="texts",
        embeddings=embedding,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    documents = retriever.get_relevant_documents(question)

    logger.info(f"Retrieved {len(documents)} documents for session: {session_id} and question: '{question}'")

    context = "\n\n".join(doc.page_content for doc in documents)
    context = context[:4000]

    logger.info(f"Loading init prompt from {INIT_PROMPT_TEMPLATE}")
    with open(INIT_PROMPT_TEMPLATE, "r", encoding="utf-8") as f:
        init_prompt_template_str = f.read()

    conversation_prompt = ChatPromptTemplate.from_template(init_prompt_template_str)
    chain: Runnable = conversation_prompt | llm | output_parser

    return StreamingResponse(
        stream_and_store_answer(chain, {"context": context, "question": question}, session_id, question),
        media_type="text/plain"
    )


# Endpoint for conversation with text/plain stream
@app.post("/conversation")
async def conversation(input: ConversationInput):
    session_id = input.session_id
    question = input.question

    # Load conversation history from Redis
    history = await get_conversation(session_id)
    previous_turns = "\n".join([
        f"Q: {turn['question']}\nA: {turn['answer']}"
        for turn in history
    ])

    # Qdrant retriever
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vectorstore = Qdrant(
        client=client,
        collection_name="texts",
        embeddings=embedding,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    documents = retriever.get_relevant_documents(question)
    logger.debug(f"Retrieved {len(documents)} documents from Qdrant for question: '{question}'")

    context = "\n\n".join(doc.page_content for doc in documents)
    context = context[:4000]

    logger.info(f"Loading conversation prompt from {PROMPT_TEMPLATE}")
    with open(PROMPT_TEMPLATE, "r", encoding="utf-8") as f:
        prompt_template_str = f.read()

    conversation_prompt = ChatPromptTemplate.from_template(prompt_template_str)
    chain: Runnable = conversation_prompt | llm | output_parser

    return StreamingResponse(
        stream_and_store_answer(
            chain,
            {"previous": previous_turns, "context": context, "question": question},
            session_id,
            question
        ),
        media_type="text/plain"
    )


# Endpoint to retrieve conversation history
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    history = await get_conversation(session_id)
    return {"session_id": session_id, "history": history}


# New conversation (delete history)
@app.post("/new-conversation")
async def new_conversation(input: SessionInput):
    await delete_conversation(input.session_id)
    return {"status": "ok", "message": f"Conversation history cleared for session: {input.session_id}"}

# Reset vector store
@app.post("/reset-vector-store")
async def reset_vector_store():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if client.collection_exists("texts"):
        client.delete_collection("texts")
        logger.warning("Qdrant 'texts' collection deleted.")

    create_qdrant_collection()
    logger.info("Qdrant 'texts' collection recreated.")

    return {"status": "ok", "message": "Vector store has been reset."}


# Get sessions
@app.get("/sessions")
async def list_sessions():
    keys = await redis_client.keys("*")
    return {"sessions": keys}

# Load documents
@app.post("/load-documents")
async def load_documents():
    # Csak .md fájlokat töltünk be
    documents = []
    for filename in os.listdir("data"):
        if filename.endswith(".md"):
            logger.info(f"Loading document: {filename}")
            loader = TextLoader(f"data/{filename}")
            documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            logger.info(f"Loading PDF: {filename}")
            loader = PyPDFLoader(f"data/{filename}")
            pdf_pages = loader.load()
            documents.extend(pdf_pages)

    if not documents:
        return {"status": "no markdown files found"}

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    # Qdrant collection creation
    create_qdrant_collection()

    # Upload documents
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vectorstore = Qdrant(
        client=client,
        collection_name="texts",
        embeddings=embedding,
    )
    vectorstore.add_documents(split_docs)

    return {
        "status": "ok",
        "documents_loaded": len(documents),
        "chunks_uploaded": len(split_docs),
        "sample_filename": documents[0].metadata['source'] if documents else "n/a",
        "sample_content": documents[0].page_content[:200] if documents else "n/a"
    }


