import os
from huggingface_hub import HfFolder, Repository, hf_hub_download
from dotenv import load_dotenv

# HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

load_dotenv()

# Set the OpenAI API key
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

if not HUGGING_FACE_API_KEY:
    raise ValueError("Missing HUGGING_FACE_API_KEY. Set it as an environment variable.")

model_id = "microsoft/Phi-3-mini-4k-instruct"

filenames = [
    ".gitattributes",
    "CODE_OF_CONDUCT.md",
    "LICENSE",
    "NOTICE.md",
    "README.md",
    "SECURITY.md",
    "added_tokens.json",
    "config.json",
    "configuration_phi3.py",
    "generation_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "modeling_phi3.py",
    "sample_finetune.py",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json"
]

for file in filenames:
    download_model_path = hf_hub_download(repo_id=model_id, filename=file, token= HUGGING_FACE_API_KEY)
    print(f"Downloaded {file} to {download_model_path}")
    


