import os
import transformers
import requests
from PIL import Image
import torch
import pandas as pd
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import accelerate
from huggingface_hub import HfFolder, Repository, hf_hub_download
from dotenv import load_dotenv

# HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

load_dotenv()

# Set the OpenAI API key
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

if not HUGGING_FACE_API_KEY:
    raise ValueError("Missing HUGGING_FACE_API_KEY. Set it as an environment variable.")

model_id = "meta-llama/Llama-3.2-3B-Instruct"

filenames = [
    ".gitattributes",
    "LICENSE.txt",
    "README.md",
    "USE_POLICY.md",
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json"
]

for file in filenames:
    download_model_path = hf_hub_download(repo_id=model_id, filename=file, token= HUGGING_FACE_API_KEY)
    print(f"Downloaded {file} to {download_model_path}")