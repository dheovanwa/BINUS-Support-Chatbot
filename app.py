from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import torch
import glob
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


if os.path.isdir('embeddings') == False:
    kb_dir = "kb"

    all_docs = []
    files = glob.glob(os.path.join(kb_dir, '*.jsonl'))

    for f in files:
        loader = JSONLoader(
            file_path=f,
            jq_schema='.content',
            text_content=False,
            json_lines=True,
        )

        loaded_docs = loader.load()
        all_docs.extend(loaded_docs)
        print(f"Loaded {len(loaded_docs)} documents from {f}")

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = text_splitter.split_documents(all_docs)

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": "cpu"}
    )

    db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory='embeddings'
    )
else:
    db = Chroma(
        persist_directory='embeddings'
    )

retriever = db.as_retriever()
template = """
Answer the question in Indonesia Language, except if the user asks in English.
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
print("Loading Qwen 3 model directly using Transformers...")
model_id = "Qwen/Qwen2-7B-Instruct" # ID model di Hugging Face Hub

# Buat pipeline text-generation
hf_pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"dtype": torch.bfloat16}, # Mengurangi penggunaan memori
    device_map="auto", # Otomatis menggunakan GPU jika ada
)

# Bungkus pipeline agar kompatibel dengan LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)
print("Llama 3 model loaded successfully.")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm # <-- Gunakan model LLM yang baru
    | StrOutputParser()
)
print("RAG Chain initialized successfully. The application is ready.")

app = FastAPI()
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    if not chain:
         return {"error": "RAG chain not initialized"}
    try:
        response = chain.invoke(request.message)
        return {"response": response}
    except Exception as e:
        print(f"Error during chain invocation: {e}")
        return {"error": "Failed to get a response from the chatbot."}

@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running."}