# import numpy as np
# import pandas as pd
# import os
# from tqdm import tqdm
# import bitsandbytes as bnb
# import torch
# import torch.nn as nn
# import transformers
# from datasets import Dataset
# from peft import LoraConfig, PeftConfig
# from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
# # from trl import setup_chat_format
# from transformers import (AutoModelForCausalLM, 
#                           AutoTokenizer, 
#                           BitsAndBytesConfig, 
#                           TrainingArguments, 
#                           pipeline, 
#                           logging)
# from sklearn.metrics import (accuracy_score, 
#                              classification_report, 
#                              confusion_matrix)
# from datetime import datetime


# labels = [    
#     "Judicial Accountability and Policy Demands",            
#     "Public Safety",                                         
#     "Socioeconomic Privilege",                               
#     "Victim Sympathy",                                       
#     "Anger or Outrage",                                      
#     "Irrelevant/General Comments",                           
#     "Views on Similar Cases in the Past"                     
# ]

# # from sklearn.model_selection import train_test_split
# def generate_prompt(data_point):
#     return f"""
#   ### Instruction:
#   Classify the comment into {', '.join(labels)}
#   Return the answer as the corresponding label.
  
#   ### Text: {data_point["text"]}
#   ### Answer: {data_point["Topic_Label"]}""".strip()

# def generate_test_prompt(data_point):
#     return f"""
#   ### Instruction:
#   Classify the comment into {', '.join(labels)}
#   Return the answer as the corresponding label.
  
#   ### Text: {data_point["text"]}
#   ### Answer:""".strip()
  
# response_template = " ### Answer:"
      
# train_df = pd.read_csv("/home/sd3528/hetav-2/data/margin/train_minority_sampling_w_topic.csv")
# val_df = pd.read_csv("/home/sd3528/hetav-2/data/margin/valid_minority_sampling_w_topic.csv")
# test_df = pd.read_csv("/home/sd3528/hetav-2/data/margin/valid_minority_sampling_w_topic.csv")

# train_df.columns = train_df.columns.str.strip()  
# val_df.columns = val_df.columns.str.strip()

# train_df["text"] = train_df["text"].fillna("").astype(str)
# val_df["text"] = val_df["text"].fillna("").astype(str)
# test_df["text"] = test_df["text"].fillna("").astype(str)
# train_df['Topic_Label'] = train_df['Topic_Label'].apply(lambda x: x.lower())
# val_df['Topic_Label'] = val_df['Topic_Label'].apply(lambda x: x.lower())
# test_df['Topic_Label'] = test_df['Topic_Label'].apply(lambda x: x.lower())

# train_df['prompt'] = train_df.apply(generate_prompt, axis=1)
# val_df['prompt'] = val_df.apply(generate_test_prompt, axis=1)
# test_df['prompt'] = test_df.apply(generate_test_prompt, axis=1)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=False,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="float16",
# )


# RUN_NAME = "llama3-8b-qlora-prompt-minority-sampling"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     # torch_dtype="float16",
#     quantization_config=bnb_config, 
# )

# model.config.use_cache = False
# model.config.pretraining_tp = 1

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenizer.pad_token_id = tokenizer.eos_token_id

# def predict(test, model, tokenizer):
#     y_pred = []
#     categories = [    
#     "Judicial Accountability and Policy Demands",            
#     "Public Safety",                                         
#     "Socioeconomic Privilege",                               
#     "Victim Sympathy",                                       
#     "Anger or Outrage",                                      
#     "Irrelevant/General Comments",                           
#     "Views on Similar Cases in the Past"                     
#     ]
    
#     for i in tqdm(range(len(test))):
#         prompt = test.iloc[i]["prompt"]
#         pipe = pipeline(task="text-generation", 
#                         model=model, 
#                         tokenizer=tokenizer, 
#                         max_new_tokens=2, 
#                         temperature=0.1)
        
#         result = pipe(prompt)
#         answer = result[0]['generated_text'].split("label:")[-1].strip()
        
#         # Determine the predicted category
#         for category in categories:
#             if category.lower() in answer.lower():
#                 y_pred.append(category)
#                 break
#         else:
#             y_pred.append("none")
    
#     return y_pred


# def find_all_linear_names(model):
#     cls = bnb.nn.Linear4bit
#     lora_module_names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, cls):
#             names = name.split('.')
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])
#     if 'lm_head' in lora_module_names:  # needed for 16 bit
#         lora_module_names.remove('lm_head')
#     return list(lora_module_names)
  
# modules = find_all_linear_names(model)
# output_dir="/home/sd3528/hetav-2/experiments/"+ RUN_NAME 

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# peft_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=modules,
# )

# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
# training_arguments = SFTConfig(
#     output_dir=output_dir,                    # directory to save and repository id
#     num_train_epochs=5,                       # number of training epochs
#     per_device_train_batch_size=1,            # batch size per device during training
#     gradient_accumulation_steps=8,            # number of steps before performing a backward/update pass
#     gradient_checkpointing=True,              # use gradient checkpointing to save memory
#     optim="paged_adamw_32bit",
#     logging_steps=1,                         
#     learning_rate=2e-4,                       # learning rate, based on QLoRA paper
#     weight_decay=0.001,
#     fp16=True,
#     bf16=False,
#     max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
#     max_steps=-1,
#     warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
#     group_by_length=False,
#     lr_scheduler_type="cosine",               # use cosine learning rate scheduler
#     report_to="wandb",                  # report metrics to w&b
#     # evalution_strategy="steps",              # save checkpoint every epoch
#     save_strategy="epoch",                   # save checkpoint every epoch
#     # packing=False,
#     max_seq_length=512,
#     dataset_text_field="prompt",
#     eval_steps = 0.2,
#     run_name=RUN_NAME+str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
# )

# train_data = Dataset.from_pandas(train_df[['prompt']])
# eval_data = Dataset.from_pandas(val_df[['prompt']])

# trainer = SFTTrainer(
#     model=model,
#     args=training_arguments,
#     train_dataset=train_data,
#     eval_dataset=eval_data,
#     peft_config=peft_config,
#     data_collator=collator,
# )


# trainer.train()

# trainer.save_model(output_dir)
# tokenizer.save_pretrained(output_dir)

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import bitsandbytes as bnb

from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from accelerate import dispatch_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Label categories
labels = [    
    "Judicial Accountability and Policy Demands",            
    "Public Safety",                                         
    "Socioeconomic Privilege",                               
    "Victim Sympathy",                                       
    "Anger or Outrage",                                      
    "Irrelevant/General Comments",                           
    "Views on Similar Cases in the Past"                     
]

# Prompt/completion splitter
def split_prompt_completion(data_point):
    prompt = f"""
### Instruction:
Classify the comment into {', '.join(labels)}
Return the answer as the corresponding label.

### Text: {data_point["text"]}
### Answer:""".strip()

    completion = f" {data_point['Topic_Label']}".strip()
    return pd.Series([prompt, completion])

# Load CSVs
train_df = pd.read_csv("/home/sd3528/hetav-2/data/margin/train_minority_sampling_w_topic.csv")
val_df = pd.read_csv("/home/sd3528/hetav-2/data/margin/valid_minority_sampling_w_topic.csv")
test_df = pd.read_csv("/home/sd3528/hetav-2/data/margin/valid_minority_sampling_w_topic.csv")

# Preprocessing
for df in [train_df, val_df, test_df]:
    df.columns = df.columns.str.strip()
    df["text"] = df["text"].fillna("").astype(str)
    df["Topic_Label"] = df["Topic_Label"].str.lower()

# Generate prompt and completion
train_df[["prompt", "completion"]] = train_df.apply(split_prompt_completion, axis=1)
val_df[["prompt", "completion"]] = val_df.apply(split_prompt_completion, axis=1)
test_df[["prompt", "completion"]] = test_df.apply(split_prompt_completion, axis=1)

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

# Load model
RUN_NAME = "llama3-8b-qlora-prompt-minority-sampling"
model_name = "meta-llama/Meta-Llama-3-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Utility: Find target modules for LoRA
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            parts = name.split('.')
            lora_module_names.add(parts[-1])
    lora_module_names.discard("lm_head")
    return list(lora_module_names)

modules = find_all_linear_names(model)

# model = dispatch_model(model, device_map="auto")

# Output directory
output_dir = f"/home/sd3528/hetav-2/experiments/{RUN_NAME}"
os.makedirs(output_dir, exist_ok=True)

# LoRA config
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,
)

# Tokenization collator
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Trainer config
training_arguments = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="wandb",
    save_strategy="epoch",
    max_seq_length=512,
    dataset_text_field="prompt",
    eval_steps=0.2,
    run_name=RUN_NAME + datetime.now().strftime("_%Y-%m-%d_%H-%M-%S"),
)

def tokenize_prompt_completion(example):
    return tokenizer(
        example["prompt"] + example["completion"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    

# Dataset conversion
train_data = Dataset.from_pandas(train_df[["prompt", "completion"]])
eval_data = Dataset.from_pandas(val_df[["prompt", "completion"]])

train_data = Dataset.from_pandas(train_df[["prompt", "completion"]])
eval_data = Dataset.from_pandas(val_df[["prompt", "completion"]])

train_data = train_data.map(tokenize_prompt_completion)
eval_data = eval_data.map(tokenize_prompt_completion)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    data_collator=collator,    
)

# Train
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)