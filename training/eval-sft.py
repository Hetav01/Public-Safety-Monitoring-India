import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Constants
MODEL_DIR = "/home/sd3528/hetav-2/experiments/llama3-8b-qlora-prompt-minority-sampling"  # Path to fine-tuned model
TEST_CSV = "/home/sd3528/hetav-2/data/margin/valid_minority_sampling_w_topic.csv"

# Label categories (should match training)
labels = [    
    "Judicial Accountability and Policy Demands",            
    "Public Safety",                                         
    "Socioeconomic Privilege",                               
    "Victim Sympathy",                                       
    "Anger or Outrage",                                      
    "Irrelevant/General Comments",                           
    "Views on Similar Cases in the Past"                     
]

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype=torch.float16)

# Load test data
df = pd.read_csv(TEST_CSV)
df.columns = df.columns.str.strip()
df["text"] = df["text"].fillna("").astype(str)
df["Topic_Label"] = df["Topic_Label"].str.lower()

# Prompt builder
def make_prompt(text):
    return f"""
### Instruction:
Classify the comment into {', '.join(labels)}
Return the answer as the corresponding label.

### Text: {text}
### Answer:""".strip()

df["prompt"] = df["text"].apply(make_prompt)

# Inference
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)#, device=0 if device == "cuda" else -1)

preds = []
for prompt in tqdm(df["prompt"].tolist(), desc="Generating predictions"):
    output = pipe(prompt, max_new_tokens=20, do_sample=False)[0]["generated_text"]
    answer = output.split("### Answer:")[-1].strip().split("\n")[0]
    preds.append(answer.lower())

# Evaluation
true_labels = df["Topic_Label"].tolist()

print("\nAccuracy:", accuracy_score(true_labels, preds))
print("\nClassification Report:\n", classification_report(true_labels, preds, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(true_labels, preds))