import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Filter out warnings
warnings.filterwarnings('ignore')

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="mps",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Load the data
df = pd.read_csv("/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/LLMsTM/unlabeled_sampled.csv")
df["label"] = ""

# Define the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Define the generation arguments
generation_args = {
    "max_new_tokens": 1500,
    "temperature": 0.7,
}

# Function to process a batch of comments
def process_batch(batch):
    results = []
    for text in batch:
        messages = [
            {"role": "system", "content": "You are a helpful, respectful and honest assistant for classifying my comments into a set of predefined labels. This is for a research project, so accuracy matters the most!"},
            {"role": "user", "content": f"""
                I have a certain comment: {text}.
                This comment is a YouTube video comment that talks about a certain accident that took place in India. Now I also have a set of 7 predefined labels/topics that this comment should be classified into.
                The labels are:
                1 - Judicial Accountability and Policy Demands: Comments discussing the legal outcomes, privilege in the judiciary, or the actions of the court.
                2 - Public Safety: Concerns raised about road safety, reckless driving, or preventive measures.
                3 - Socioeconomic Privilege: Comments highlighting class dynamics, privilege, or inequality in legal consequences.
                4 - Victim Sympathy: Empathy expressed towards the victims and their families.
                5 - Anger or Outrage: Expressions of frustration, dissatisfaction, or anger about the incident or its handling.
                6 - Irrelevant/General Comments: Off-topic remarks, emojis, or comments with little context.
                7 - Views on Similar Cases in the Past: Talks about another case in the past.
                
                As an Example: if the Input: "The court must & should give capital punishment then only things will come to order", the Output: 1.
                
                The reason behind this is that the comment is about the court's decision and the need for capital punishment, which falls under Judicial Accountability and Policy Demands.
                If the Input: ":fire::victory_hand::OK_hand::hundred_points::crossed_fingers::thumbs_up:", the Output: 6.
                If the Input: ":red_heart::smiling_face:", the Output: 6.
                The reason behind this is that the comment is just emojis, which falls under Irrelevant/General Comments.
                If the Input: "Does anyone know ""Cameron Herrin"" case ?????", the Output: 7.
                If the Input: "Money can buy law and everything", the Output: 3.
                If the Input: "Ye to choti accident thi sukar hai bike mai silence nahi laga tha varna jail ho jati", the Output: 2.
                If the Input: "Justice for Ashwini and Aneesh", the Output: 4.
                If the Input: "Judiciary,,,,,,moye moye ho gya", the Output: 5.
                Based on this information, classify the comment into one of the 7 labels.
                Note: There will be many irrelevant comments, so be sure and try to classify them into 6. Be strict about the irrelevant comments. If there's a small sign of irrelevance, classify it into 6.
                You only need to return the label number and nothing else.
            """},
        ]
        
        output = pipe(messages, **generation_args)
        generated_text = output[0]["generated_text"]
        label = generated_text.strip()
        results.append(label)
    return results

# Process the comments in batches using parallel processing
batch_size = 100
num_batches = len(df) // batch_size + 1

with ThreadPoolExecutor() as executor:
    futures = []
    for i in range(num_batches):
        batch = df["textCleaned"][i * batch_size:(i + 1) * batch_size].tolist()
        futures.append(executor.submit(process_batch, batch))
    
    for future in as_completed(futures):
        batch_results = future.result()
        start_idx = futures.index(future) * batch_size
        df.loc[start_idx:start_idx + len(batch_results) - 1, "label"] = batch_results

# Save the labeled data to a CSV file
df.to_csv("/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/LLMsTM/labeled_sampled.csv", index=False)