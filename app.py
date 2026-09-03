from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import re
from pathlib import Path
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import time
import socket
import os

# =========================================================
# FastAPI Application
# =========================================================
app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)

# =========================================================
# Template Configuration
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(
    directory=str(BASE_DIR)
)

# =========================================================
# Hugging Face Configuration
# =========================================================
# Store the token in an environment variable; never commit secrets.
HF_TOKEN = os.getenv("HF_TOKEN")

# =========================================================
# Request Model
# =========================================================
class DialogueInput(BaseModel):
    dialogue: str

# =========================================================
# Text Cleaning
# =========================================================
def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

# =========================================================
# Test Network Connectivity
# =========================================================
def test_network_connectivity():
    """Test if we can reach Hugging Face API"""
    try:
        # Try multiple DNS servers
        hostnames = ['api-inference.huggingface.co', 'huggingface.co', 'cdn-lfs.huggingface.co']
        for hostname in hostnames:
            try:
                socket.gethostbyname(hostname)
                print(f"DNS resolution successful for {hostname}")
                return True
            except socket.gaierror:
                continue
        
        # If all fail, try direct HTTP request
        try:
            response = requests.get('https://huggingface.co', timeout=5)
            if response.status_code == 200:
                print("HTTP connection to huggingface.co successful")
                return True
        except:
            pass
            
        print("DNS resolution failed - check your internet connection")
        return False
    except Exception as e:
        print(f"Network test error: {e}")
        return False

# =========================================================
# Text Summarization with Multiple Fallbacks
# =========================================================
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)
    
    # Test network connectivity first
    if not test_network_connectivity():
        return "Network error: Cannot connect to Hugging Face API. Please check your internet connection."
    
    # Try multiple models and endpoints as fallbacks
    models = [
        "t5-small",
        "t5-base",
        "google/t5-small-ssm"
    ]
    
    for model in models:
        try:
            result = call_huggingface_api(dialogue, model)
            if result and "Error" not in result and "Unable" not in result:
                return result
        except Exception as e:
            print(f"Model {model} failed: {e}")
            continue
    
    return "Unable to generate summary after trying multiple models."

def call_huggingface_api(dialogue: str, model: str) -> str:
    """Call Hugging Face API with authentication"""
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    # Create a session with retry logic
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504, 429, 401, 403],
        allowed_methods=["POST"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        response = session.post(
            API_URL,
            headers=headers,
            json={
                "inputs": "summarize: " + dialogue,
                "parameters": {
                    "max_length": 150,
                    "min_length": 30,
                    "length_penalty": 2.0,
                    "num_beams": 4,
                    "early_stopping": True
                }
            },
            timeout=30  # Reduced timeout
        )
        
        # Check HTTP status
        if response.status_code == 503:
            return "Model is currently loading. Please try again in a few seconds."
        
        response.raise_for_status()
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if "generated_text" in result[0]:
                return result[0]["generated_text"]
        elif isinstance(result, dict):
            if "generated_text" in result:
                return result["generated_text"]
            if "error" in result:
                return f"Hugging Face Error: {result['error']}"
        
        return "Unable to generate summary."
        
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except requests.exceptions.ConnectionError as e:
        return f"Connection error: {str(e)}"
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            return "Authentication error: Invalid API token"
        elif response.status_code == 403:
            return "Access forbidden: Check your token permissions"
        elif response.status_code == 429:
            return "Rate limit exceeded. Please wait a moment and try again."
        else:
            return f"HTTP error: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"API request failed: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# =========================================================
# Alternative: Use Local Model (No API)
# =========================================================
def summarize_dialogue_local(dialogue: str) -> str:
    """Fallback using local model if available"""
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        
        # Load model locally (first time will download)
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        dialogue = clean_data(dialogue)
        
        # Tokenize input
        inputs = tokenizer.encode(
            "summarize: " + dialogue, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        # Generate summary
        summary_ids = model.generate(
            inputs, 
            max_length=150, 
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode and return summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        
    except ImportError:
        return "Local model not available. Please install transformers and torch: pip install transformers torch"
    except Exception as e:
        return f"Local model error: {str(e)}"

# =========================================================
# API Endpoint - Summarize with Options
# =========================================================
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput, local: bool = False):
    if local:
        summary = summarize_dialogue_local(dialogue_input.dialogue)
    else:
        summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

# =========================================================
# Health Check Endpoint
# =========================================================
@app.get("/health")
async def health_check():
    """Check if the API is working"""
    connectivity = test_network_connectivity()
    token_valid = HF_TOKEN is not None and HF_TOKEN.startswith("hf_")
    
    return {
        "status": "healthy" if connectivity and token_valid else "unhealthy",
        "connectivity": connectivity,
        "token_valid": token_valid,
        "token_prefix": HF_TOKEN[:4] + "..." if HF_TOKEN else "None"
    }

# =========================================================
# Home Page
# =========================================================
@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )