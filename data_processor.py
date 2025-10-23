"""
Data processing utilities for HaleAI
Handles PDF loading, text splitting, and embeddings
"""
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from config import *


class DataProcessor:
    """Handles all data processing operations"""
    
    def __init__(self):
        self.embeddings = self._load_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    
    def _load_embeddings(self):
        """Load HuggingFace embeddings model"""
        print("⏳ Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        print("✅ Embeddings model loaded")
        return embeddings
    
    def load_pdf_files(self, data_dir: str) -> List[Document]:
        """Load all PDF files from directory"""
        print(f"📂 Loading PDFs from {data_dir}...")
        loader = DirectoryLoader(
            data_dir,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        print(f"📄 Loaded {len(documents)} documents")
        return documents
    
    def filter_metadata(self, docs: List[Document]) -> List[Document]:
        """Keep only essential metadata"""
        minimal_docs = []
        for doc in docs:
            minimal_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={"source": doc.metadata.get("source", "unknown")}
                )
            )
        return minimal_docs
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        print("✂️ Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(docs)
        print(f"📝 Created {len(chunks)} chunks")
        return chunks
    
    def process_documents(self, data_dir: str) -> List[Document]:
        """Complete document processing pipeline"""
        docs = self.load_pdf_files(data_dir)
        docs = self.filter_metadata(docs)
        chunks = self.split_documents(docs)
        return chunks