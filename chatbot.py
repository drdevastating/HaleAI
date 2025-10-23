"""
Main HaleAI Medical Chatbot class
Orchestrates all components for RAG-based medical Q&A
"""
from typing import Dict, List
from data_processor import DataProcessor
from vector_store import VectorStoreManager
from llm_handler import LLMHandler
from langchain.schema import Document


class HaleAI:
    """HaleAI Medical Chatbot with RAG capabilities"""
    
    def __init__(self):
        print("\n🏥 Initializing HaleAI Medical Chatbot...")
        print("=" * 60)
        
        self.data_processor = DataProcessor()
        self.vector_store = VectorStoreManager(self.data_processor.embeddings)
        self.llm = LLMHandler()
        
        print("=" * 60)
        print("✅ HaleAI initialized successfully!\n")
    
    def setup_knowledge_base(self, data_dir: str):
        """
        Initial setup: Process documents and load to Pinecone
        Only run this once when setting up the chatbot
        """
        print("\n📚 Setting up knowledge base...")
        print("=" * 60)
        
        # Process documents
        chunks = self.data_processor.process_documents(data_dir)
        
        # Load to Pinecone
        self.vector_store.load_documents(chunks)
        
        print("=" * 60)
        print("✅ Knowledge base setup complete!\n")
    
    def connect(self):
        """Connect to existing knowledge base"""
        print("\n🔗 Connecting to knowledge base...")
        self.vector_store.connect_to_existing_index()
        print("✅ Connected!\n")
    
    def query(self, user_question: str) -> Dict:
        """
        Process a user query and generate a response
        
        Args:
            user_question: The user's medical question
            
        Returns:
            Dictionary containing the answer, sources, and metadata
        """
        try:
            print(f"\n💬 Query: {user_question}")
            print("-" * 60)
            
            # Step 1: Retrieve relevant documents
            print("🔍 Searching knowledge base...")
            retrieved_docs = self.vector_store.retrieve_documents(user_question)
            print(f"📚 Found {len(retrieved_docs)} relevant sources")
            
            # Step 2: Format context
            context = self.llm.format_context(retrieved_docs)
            
            # Step 3: Create prompt
            prompt = self.llm.create_prompt(user_question, context)
            
            # Step 4: Generate response
            print("🤖 Generating answer...")
            raw_response = self.llm.generate(prompt)
            
            # Step 5: Extract clean answer
            answer = self.llm.extract_answer(raw_response, prompt)
            
            print("✅ Answer generated")
            print("-" * 60)
            
            return {
                "question": user_question,
                "answer": answer,
                "sources": retrieved_docs,
                "num_sources": len(retrieved_docs),
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {
                "question": user_question,
                "answer": "I apologize, but I encountered an error processing your question. Please try rephrasing it.",
                "sources": [],
                "num_sources": 0,
                "status": "error",
                "error": str(e)
            }
    
    def chat(self):
        """Interactive chat interface"""
        print("\n" + "=" * 60)
        print("🏥 HaleAI Medical Chatbot - Interactive Mode")
        print("=" * 60)
        print("Ask me any medical questions. Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thank you for using HaleAI. Stay healthy!")
                    break
                
                response = self.query(user_input)
                print(f"\n🤖 HaleAI: {response['answer']}\n")
                print(f"📊 Based on {response['num_sources']} medical sources\n")
                print("-" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
    
    def batch_test(self, questions: List[str]) -> List[Dict]:
        """Test multiple questions at once"""
        print("\n🧪 Running batch test...")
        print("=" * 60 + "\n")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}]")
            response = self.query(question)
            results.append(response)
            print()
        
        print("=" * 60)
        print("✅ Batch test complete!\n")
        return results