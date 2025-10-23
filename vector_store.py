"""
Vector store management for HaleAI
Handles Pinecone index creation and retrieval
"""
from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from config import *


class VectorStoreManager:
    """Manages Pinecone vector store operations"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self._ensure_index_exists()
        self.vectorstore = None
        self.retriever = None
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        if not self.pc.has_index(self.index_name):
            print(f"🔨 Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=PINECONE_DIMENSION,
                metric=PINECONE_METRIC,
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
            )
            print(f"✅ Index created: {self.index_name}")
        else:
            print(f"✅ Index {self.index_name} already exists")
    
    def load_documents(self, documents: List[Document]):
        """Load documents into Pinecone (run once for initial setup)"""
        print("📤 Uploading documents to Pinecone...")
        self.vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        print("✅ Documents uploaded successfully")
        self._create_retriever()
    
    def connect_to_existing_index(self):
        """Connect to existing Pinecone index"""
        print(f"🔗 Connecting to existing index: {self.index_name}")
        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        print("✅ Connected to vector store")
        self._create_retriever()
    
    def _create_retriever(self):
        """Create retriever from vector store"""
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K}
        )
        print(f"✅ Retriever created (k={RETRIEVER_K})")
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query"""
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call connect_to_existing_index() first.")
        return self.retriever.invoke(query)
    
    def test_retrieval(self, query: str = "What is acne?"):
        """Test retrieval with a sample query"""
        print(f"\n🧪 Testing retrieval with query: '{query}'")
        docs = self.retrieve_documents(query)
        print(f"📚 Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            print(f"  Doc {i}: {doc.page_content[:80]}...")
        return docs