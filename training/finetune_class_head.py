# import pandas as pd
# from transformers import LlamaTokenizer, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model, PeftModel, TaskType
# from datasets import Dataset
# from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
# from trl import SFTTrainer
# import torch
# from accelerate import Accelerator
# from transformers import LlamaForSequenceClassification
# from datetime import datetime
# import os

# accelerator = Accelerator()

# # Load the dataset from CSV files
# train_df = pd.read_csv("/home/sd3528/hetav-2/data/train.csv")
# test_df = pd.read_csv("/home/sd3528/hetav-2/data/valid.csv")

# train_df.columns = train_df.columns.str.strip()  
# test_df.columns = test_df.columns.str.strip()

# train_df["text"] = train_df["text"].fillna("").astype(str)
# test_df["text"] = test_df["text"].fillna("").astype(str)

# train_df = train_df.dropna(subset=["label"])
# test_df = test_df.dropna(subset=["label"])

# # label_map = {"Negative": 0, "Neutral": 1, "Positive": 2, "negative": 0, "neutral": 1, "positive": 2}
# train_df["label"] = train_df["label"].astype(int)
# test_df["label"] = test_df["label"].astype(int)

# train_dataset = Dataset.from_pandas(train_df)
# test_dataset = Dataset.from_pandas(test_df)

# RUN_NAME = "llama3-8b-qlora-v2"

# # Load the LLaMA tokenizer and model
# # model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = 'mistral'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token


# # Configure LoRA
# lora_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS,  # Sequence classification task
#     inference_mode=False,        # Enable training mode
#     r=8,                         # LoRA rank
#     lora_alpha=64,               # LoRA scaling factor
#     lora_dropout=0.1             # LoRA dropout rate
# )

# compute_dtype = getattr(torch, "float16")
# use_4bit = True
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=use_4bit, # Load model in 4bit
#     bnb_4bit_quant_type="nf4", # Use 4bit quantization. NormalFloat4 
#     bnb_4bit_compute_dtype=compute_dtype, # Use float16 for computation
#     bnb_4bit_use_double_quant=False, # Use double quantization
# )

# model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=7, quantization_config=bnb_config)
# model.resize_token_embeddings(len(tokenizer))
# model.config.pad_token_id = model.config.eos_token_id
# model.config.num_labels = 7

# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, lora_config)


# # Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
# tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
# tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels") 

# output_path = f'/home/sd3528/hetav-2/experiments/{RUN_NAME}/'

# if not os.path.exists(output_path):
#     os.makedirs(output_path)
#     print(f"Directory {output_path} created.")
    
# # Define training arguments
# training_args = TrainingArguments(
#     output_dir=output_path+"results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-4,           # Higher learning rate for LoRA
#     per_device_train_batch_size=2, # Larger batch size
#     per_device_eval_batch_size=2,    
#     gradient_accumulation_steps=4,
#     num_train_epochs=10,
#     fp16=True,
#     logging_dir=f"{output_path}/logs",
#     logging_steps=1,
#     save_strategy="epoch",
#     save_total_limit=2,
#     run_name=RUN_NAME+str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
# )

# # # Define a Trainer instance
# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=tokenized_train_dataset,
# #     eval_dataset=tokenized_test_dataset,
# #     tokenizer=tokenizer
# # )

# trainer = Trainer(
#     model=model,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_test_dataset,
#     # peft_config=lora_config,    
#     tokenizer=tokenizer,
#     args=training_args,            
# )

# # Train the model with LoRA
# trainer.train()

# # Save the fine-tuned LoRA model
# trainer.save_model(output_path+"lora_model")

# print("Fine-tuning completed with Quanr and model saved!")

import pandas as pd
from transformers import LlamaTokenizer, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import SFTTrainer
import torch
from accelerate import Accelerator
from transformers import LlamaForSequenceClassification
from datetime import datetime

accelerator = Accelerator()

# Load the dataset from CSV files
train_df = pd.read_csv("/home/sd3528/hetav-2/data/original/train.csv")
test_df = pd.read_csv("/home/sd3528/hetav-2/data/original/test.csv")

train_df.columns = train_df.columns.str.strip()  
test_df.columns = test_df.columns.str.strip()

train_df["text"] = train_df["text"].fillna("").astype(str)
test_df["text"] = test_df["text"].fillna("").astype(str)

train_df = train_df.dropna(subset=["label"])
test_df = test_df.dropna(subset=["label"])

# label_map = {"Negative": 0, "Neutral": 1, "Positive": 2, "negative": 0, "neutral": 1, "positive": 2}
# train_df["label"] = train_df["label"].map(label_map).astype(int)
# test_df["label"] = test_df["label"].map(label_map).astype(int)
train_df["label"] = train_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)

print("Train label stats:", train_df['label'].unique())
print("Test label stats:", test_df['label'].unique())

assert pd.api.types.is_integer_dtype(train_df['label'])

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# RUN_NAME = "mistral-7b-qlora-v2"
RUN_NAME = "llama3-8b-qlora-org-vfinal"

# Load the LLaMA tokenizer and model
model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = 'mistralai/Mistral-7B-v0.3'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification task
    inference_mode=False,        # Enable training mode
    r=8,                         # LoRA rank
    lora_alpha=64,               # LoRA scaling factor
    lora_dropout=0.1             # LoRA dropout rate
)

compute_dtype = getattr(torch, "float16")
use_4bit = True
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit, # Load model in 4bit
    bnb_4bit_quant_type="nf4", # Use 4bit quantization. NormalFloat4 
    bnb_4bit_compute_dtype=compute_dtype, # Use float16 for computation
    bnb_4bit_use_double_quant=False, # Use double quantization
)

model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=7, quantization_config=bnb_config, device_map = "auto")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model.config.num_labels = 7

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels") 

# Define training arguments
training_args = TrainingArguments(
    output_dir=f"/home/sd3528/hetav-2/experiments/{RUN_NAME}/results",
    # evaluation_strategy="epoch",
    learning_rate=2e-4,           # Higher learning rate for LoRA
    per_device_train_batch_size=2, # Larger batch size
    per_device_eval_batch_size=2,    
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    # fp16=True,
    logging_dir=f"/home/sd3528/hetav-2/experiments/{RUN_NAME}/logs",
    logging_steps=2,
    save_strategy="epoch",
    save_total_limit=2,
    run_name=RUN_NAME+str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
)

# # Define a Trainer instance
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_test_dataset,
#     tokenizer=tokenizer
# )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    # peft_config=lora_config,    
    tokenizer=tokenizer,
    args=training_args,            
)

# Train the model with LoRA
trainer.train()

# Save the fine-tuned LoRA model
trainer.save_model(f"/home/sd3528/hetav-2/experiments/{RUN_NAME}/model")

print("Fine-tuning completed with Quanr and model saved!")