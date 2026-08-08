import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PDF_PATH = BASE_DIR / "ESI-Handbook-5th-Edition-3-2023.pdf"
DEFAULT_PERSIST_DIR = BASE_DIR / "db" / "chroma_db"

EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)

def load_pdf(pdf_path=DEFAULT_PDF_PATH):
    path = Path(pdf_path)
    if not path.exists():
        pdf_files = list(BASE_DIR.glob("*.pdf"))
        if pdf_files:
            path = pdf_files[0]
        else:
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    print(f"Loading PDF from {path}...")
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")
    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=150):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")
    return chunks

def create_vector_store(chunks, persist_directory=DEFAULT_PERSIST_DIR):
    persist_path = Path(persist_directory)
    if persist_path.exists():
        shutil.rmtree(persist_path)
        print("Cleared existing vector store.")

    print("Creating embeddings and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDINGS_MODEL,
        persist_directory=str(persist_path),
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Successfully created vector store at '{persist_path}'.")
    return vectorstore

def main():
    documents = load_pdf()
    chunks = split_documents(documents)
    create_vector_store(chunks)

if __name__ == "__main__":
    main()