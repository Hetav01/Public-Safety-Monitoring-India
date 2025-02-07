# iwant to use my mac M gpu to run the model, how do i do that?
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import pandas as pd

# if torch.backends.mps.is_available():
#    mps_device = torch.device("mps")
#    x = torch.ones(1, device=mps_device)
#    print (x)
# else:
#    print ("MPS device not found.")
   
# # GPU
# start_time = time.time()

# # syncrocnize time with cpu, otherwise only time for oflaoding data to gpu would be measured
# torch.mps.synchronize()

# a = torch.ones(4000,4000, device="mps")
# for _ in range(200):
#    a +=a

# elapsed_time = time.time() - start_time
# print( "GPU Time: ", elapsed_time)


torch.random.manual_seed(0) 
model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    device_map="mps",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

df = pd.read_csv("/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/preprocessing/sample.csv")

df['textTranslated'] = ""

for i in range(len(df)):
    #select the textOriginal column from the df
    text = df["textOriginal"][i]
    
    messages = [ 
        {"role": "system", "content": "You are a helpful AI assistant."}, 
        {"role": "user", "content": f"Can you translate this text: {text} into English? You only need to return the translation and nothing else. If you encounter any emoji, demojize them immediately into a text form of what it means. Output should look like this: Translation: <translated_text>"}, 
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
    
    # Debugging: Print the output to see its structure
    # print(output)
    print("-------------------------\n")
    
    # Extract the translation part from the output
    generated_text = output[0]["generated_text"]
    
    # Find the dictionary with "role" set to "assistant" and extract the "content"
    translation = ""
    for message in generated_text:
        if message["role"] == "assistant":
            translation = message["content"].rsplit("Translation: ", 1)[-1].strip()
            break
    
    # Add the translation to the df as a new column named "textTranslated"
    df.loc[i, "textTranslated"] = translation
    
    #print the output
    print(translation)    
    print("-------------------------\n")

#save the sample translation df to a csv file
df.to_csv("/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/preprocessing/sampleNew.csv", index=False)