from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Weaviate
from langchain.embeddings import OllamaEmbeddings
import weaviate
import os

app = FastAPI()

# Load environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# Initialize LLM and embedding model
llm = ChatOllama(model="mistral", base_url=OLLAMA_BASE_URL)
embedding = OllamaEmbeddings(model="mistral", base_url=OLLAMA_BASE_URL)

# Prompt template
base_prompt = ChatPromptTemplate.from_template(
    "Answer the following question using the provided context.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}"
)

output_parser = StrOutputParser()

# Question input model
class QuestionInput(BaseModel):
    question: str

# Endpoint: first question
@app.post("/init-question")
async def init_question(input: QuestionInput):
    # Connect to Weaviate
    client = weaviate.Client(WEAVIATE_URL)

    # LangChain retriever
    vectorstore = Weaviate(
        client=client,
        index_name="texts",      # Make sure this matches your Weaviate class
        text_key="text",         # The property name in Weaviate
        embedding=embedding,
    )
    retriever = vectorstore.as_retriever()

    # Retrieve context
    documents = retriever.get_relevant_documents(input.question)
    context = "\n\n".join(doc.page_content for doc in documents)

    # Build chain
    chain: Runnable = base_prompt | llm | output_parser
    result = chain.invoke({"context": context, "question": input.question})

    return {"response": result}
