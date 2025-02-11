import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import pandas as pd
import accelerate

#using phi3 right now due to llama taking a shit ton of time.
# model_id = "meta-llama/Llama-3.2-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    device_map="mps",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

#filter out warnings
import warnings 
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

#change the path name according to your preference.
df = pd.read_csv("/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/LLMsTM/unlabeled_sampled.csv")

df["label"] = ""

for i in range(100):    # set the loop for only first 500 comments.
    text = df["textCleaned"][i]
    
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
            
            The reason behind this is that the comment is just emojis, which falls under Irrelevant/General Comments.
            
            If the Input: "Does anyone know ""Cameron Herrin"" case ?????", the Output: 7.
            
            If the Input: "Money can buy law and everything", the Output: 3.
            
            If the Input: "Ye to choti accident thi sukar hai bike mai silence nahi laga tha varna jail ho jati", the Output: 2.
            
            If the Input: "Justice for Ashwini and Aneesh", the Output: 4.
            
            If the Input: "Judiciary,,,,,,moye moye ho gya", the Output: 5.
            
            Based on this information, classify the comment into one of the 7 labels. 
            
            You only need to return the label number and nothing else.
         """}, 
    ] 
    
    pipe = pipeline( 
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
    )
    
    generation_args = {
        "max_new_tokens": 1500,
        "temperature": 0.7,
    }
    
    output = pipe(messages, **generation_args)
    # print(output)
    print("-------------------------\n")
    
    # add this label to the df in a new column named "label"
    generated_text = output[0]["generated_text"]
    
    #if the role is assistant, then add the content to the df
    for message in generated_text:
        if message["role"] == "assistant":
            df.loc[i, "label"] = message["content"]
            print(df["label"][i])
            break
    
    print("-------------------------\n")
    

#save the sample translation df to a csv file
df.to_csv("/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/LLMsTM/labeled_sampled.csv", index=False)
    
    