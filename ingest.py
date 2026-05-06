import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
DATA_DIR = "./data"
DB_DIR = "./chroma_db"

def build_vector_database():
    print("Loading documents from ./data directory...")
    # 1. Load PDFs
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()
    
    if not docs:
        print("No documents found! Please put PDFs in the 'data' folder.")
        return

    # 2. Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)

    # 3. Create Embeddings using Ollama
    print("Initializing embedding model...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Store in Chroma Vector Database
    print("Creating vector database...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"Successfully embedded {len(splits)} chunks into ChromaDB!")

if __name__ == "__main__":
    build_vector_database()