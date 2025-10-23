# HaleAI

we need to create a virtual environment as well : 
how ? 

conda create -n HaleAI python=3.11 -y
pip install -U "langchain>=0.3.7" "langchain-core>=0.3.7" "langchain-pinecone>=0.1.0" "langchain-openai>=0.2.0" packaging>=24.2

# HaleAI Medical Chatbot - Setup Guide

## Prerequisites
- Python 3.8 or higher
- Internet connection (for API calls)
- Pinecone account (free tier available)
- Hugging Face account (free)

## Step 1: Clone/Download Project
```bash
# Your project structure should look like:
# haleai/
# ├── main.py
# ├── chatbot.py
# ├── llm_handler.py
# ├── vector_store.py
# ├── data_processor.py
# ├── config.py
# ├── requirements.txt
# ├── .env
# └── data/
#     └── your_medical_pdfs.pdf
```

## Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 4: Get API Keys

### Pinecone API Key
1. Go to https://app.pinecone.io/
2. Sign up for free account
3. Create a new project
4. Copy your API key from the dashboard

### Hugging Face Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "read" permissions
3. Copy the token

## Step 5: Configure Environment Variables
Create a `.env` file in the project root:
```
PINECONE_API_KEY=your_actual_pinecone_key
HF_TOKEN=your_actual_huggingface_token
```

## Step 6: Add Medical Documents
Place your PDF files in the `data/` folder

## Step 7: Run Setup (First Time Only)
```bash
python main.py
# Select option 1 (Setup)
```

This will:
- Process your PDF documents
- Create embeddings
- Upload to Pinecone vector database

## Step 8: Start Chatting
```bash
python main.py
# Select option 2 (Chat)
```

## Alternative Free Models

If Mistral-7B-Instruct doesn't work well or is too slow, you can try these alternatives by changing `LLM_MODEL` in `config.py`:

### Option 1: Zephyr (Good for chat)
```python
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"
```

### Option 2: Llama 2 (Popular, well-tested)
```python
LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"
```
Note: May require accepting license on HuggingFace

### Option 3: Phi-2 (Smaller, faster)
```python
LLM_MODEL = "microsoft/phi-2"
```

## Troubleshooting

### "Model is loading" error
- First API call takes 20-60 seconds to load the model
- Subsequent calls are faster
- Just wait and it will work

### Rate limiting
- Free Hugging Face API has rate limits
- Wait a minute between queries if you hit limits
- Consider upgrading to Pro for unlimited access

### Out of memory
- You're using API, so no local memory issues!
- If embeddings cause issues, reduce CHUNK_SIZE in config.py

### Slow responses
- First query is slow (model loading)
- Later queries are faster
- Try smaller models like phi-2

## Usage Tips

1. **First query is slow**: The model needs to load (20-60 seconds)
2. **Be patient**: API calls take 3-10 seconds per response
3. **Rate limits**: Free tier allows ~100 requests/hour
4. **Better results**: Ask specific, clear medical questions

## Cost Comparison

| Resource | Free Tier | Cost |
|----------|-----------|------|
| Pinecone | 1 index, 100K vectors | Free forever |
| Hugging Face API | ~100 requests/hour | Free (Pro: $9/month) |
| Embeddings | Unlimited | Free (local) |

**Total Cost: $0** (with free tier limits)

## Support

If you encounter issues:
1. Check your API keys are correct
2. Ensure you have internet connection
3. Wait if model is loading
4. Try alternative models
5. Check Hugging Face model status page