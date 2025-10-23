"""
LLM handler for HaleAI
Manages Mistral model via Hugging Face API
"""
import requests
import time
from typing import List
from langchain.schema import Document
from config import *


class LLMHandler:
    """Handles LLM operations via Hugging Face API"""
    
    def __init__(self):
        self.api_url = HF_API_URL
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        self._validate_token()
    
    def _validate_token(self):
        """Validate HF token is present"""
        if not HF_TOKEN:
            raise ValueError(
                "HF_TOKEN not found in environment variables. "
                "Please add it to your .env file. "
                "Get your token from https://huggingface.co/settings/tokens"
            )
        print("✅ Hugging Face API token validated")
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate response from Hugging Face API
        
        Args:
            prompt: The input prompt
            max_retries: Number of retry attempts if model is loading
            
        Returns:
            Generated text response
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "repetition_penalty": REPETITION_PENALTY,
                "return_full_text": False
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                # Debug: Print response details
                print(f"🔍 API Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return result[0]["generated_text"]
                        return str(result)
                    except ValueError as json_error:
                        print(f"❌ JSON Parse Error. Raw response: {response.text[:200]}")
                        return f"Error: Invalid API response format. Response: {response.text[:100]}"
                
                elif response.status_code == 503:
                    # Model is loading
                    try:
                        error_data = response.json()
                        wait_time = error_data.get("estimated_time", 20)
                        print(f"⏳ Model is loading... waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    except ValueError:
                        print(f"⏳ Model is loading... waiting 20s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(20)
                        continue
                
                elif response.status_code == 401:
                    return "Error: Invalid Hugging Face token. Please check your HF_TOKEN in .env file."
                
                elif response.status_code == 404:
                    return f"Error: Model not found. Please check the model name: {LLM_MODEL}"
                
                else:
                    # Try to get error message
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", "Unknown error")
                        print(f"❌ API Error {response.status_code}: {error_msg}")
                        return f"API Error ({response.status_code}): {error_msg}"
                    except ValueError:
                        print(f"❌ API Error {response.status_code}: {response.text[:200]}")
                        return f"API Error ({response.status_code}): {response.text[:100]}"
            
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"⏱️ Request timeout, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(5)
                    continue
                return "Error: Request timed out. Please try again."
            
            except requests.exceptions.ConnectionError:
                return "Error: Connection failed. Please check your internet connection."
            
            except Exception as e:
                print(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
                return f"Error: {str(e)}"
        
        return "Error: Model is taking too long to load. Please try again in a few minutes."
    
    def format_context(self, retrieved_docs: List[Document]) -> str:
        """Format retrieved documents as context"""
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.page_content.strip()
            # Limit each source to avoid token overflow
            if len(content) > 300:
                content = content[:300] + "..."
            context_parts.append(f"[Source {i}] {content}")
        
        return "\n\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create optimized prompt for the LLM
        Format adapts based on the model being used
        """
        # Zephyr/ChatML format (works for most instruction models)
        prompt = f"""<|system|>
You are a helpful medical assistant. Answer questions based on the provided medical information. Be concise, accurate, and professional.</s>
<|user|>
Medical Information:
{context}

Question: {query}</s>
<|assistant|>"""
        
        return prompt
    
    def extract_answer(self, generated_text: str, original_prompt: str = None) -> str:
        """
        Clean and extract the answer from generated text
        
        Args:
            generated_text: The raw generated text from the API
            original_prompt: The original prompt (optional, for compatibility)
        
        Returns:
            Cleaned answer text
        """
        # The API with return_full_text=False only returns the generated part
        # So we don't need to remove the prompt
        
        # Remove instruction tags if present (supports multiple formats)
        answer = generated_text
        for tag in ["[/INST]", "<s>", "</s>", "<|assistant|>", "<|user|>", "<|system|>"]:
            answer = answer.replace(tag, "")
        answer = answer.strip()
        
        # Remove any remaining prompt artifacts
        if original_prompt and answer.startswith(original_prompt):
            answer = answer[len(original_prompt):].strip()
        
        # Split into sentences and clean
        sentences = answer.split('. ')
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short fragments
                if not sentence.endswith('.') and not sentence.endswith('?') and not sentence.endswith('!'):
                    sentence += '.'
                cleaned_sentences.append(sentence)
        
        final_answer = ' '.join(cleaned_sentences).strip()
        
        # Limit length
        if len(final_answer) > 600:
            # Cut at last complete sentence within limit
            truncated = final_answer[:600]
            last_period = truncated.rfind('.')
            if last_period > 0:
                final_answer = truncated[:last_period + 1]
            else:
                final_answer = truncated + "..."
        
        return final_answer if final_answer else "I don't have sufficient information to answer this question."