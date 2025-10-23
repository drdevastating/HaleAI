"""
Main entry point for HaleAI Medical Chatbot
Run this file to use the chatbot
"""
from chatbot import HaleAI
from config import DATA_DIR


def setup_new_chatbot():
    """
    First-time setup: Process documents and create knowledge base
    Only run this once!
    """
    print("\n🔧 FIRST-TIME SETUP MODE")
    print("This will process your medical documents and upload them to Pinecone.")
    print("This only needs to be run once.\n")
    
    confirm = input("Continue with setup? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Setup cancelled.")
        return
    
    chatbot = HaleAI()
    chatbot.setup_knowledge_base(DATA_DIR)
    
    print("\n✅ Setup complete! You can now run the chatbot in normal mode.")


def run_chatbot():
    """
    Normal mode: Connect to existing knowledge base and start chatting
    """
    chatbot = HaleAI()
    chatbot.connect()
    
    # Start interactive chat
    chatbot.chat()


def run_tests():
    """
    Test mode: Run predefined test queries
    """
    chatbot = HaleAI()
    chatbot.connect()
    
    test_questions = [
        "What is acne?",
        "What causes body pain?",
        "What are the symptoms of diabetes?",
        "How to treat a fever?",
        "What are the side effects of aspirin?"
    ]
    
    results = chatbot.batch_test(test_questions)
    
    # Print summary
    print("\n📊 TEST SUMMARY")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Q: {result['question']}")
        print(f"   A: {result['answer'][:100]}...")
        print(f"   Sources: {result['num_sources']}")


def single_query():
    """
    Single query mode: Ask one question and exit
    """
    chatbot = HaleAI()
    chatbot.connect()
    
    question = input("\n💬 Enter your medical question: ").strip()
    if question:
        response = chatbot.query(question)
        print(f"\n🤖 Answer:\n{response['answer']}\n")
        print(f"📊 Based on {response['num_sources']} sources")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🏥 HaleAI - Medical Chatbot with RAG")
    print("=" * 60)
    print("\nSelect mode:")
    print("1. Setup (first-time only)")
    print("2. Chat (interactive)")
    print("3. Test (run predefined tests)")
    print("4. Single Query")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        setup_new_chatbot()
    elif choice == "2":
        run_chatbot()
    elif choice == "3":
        run_tests()
    elif choice == "4":
        single_query()
    elif choice == "5":
        print("\n👋 Goodbye!")
    else:
        print("\n❌ Invalid choice. Please run again.")