import pandas as pd
import numpy as np
import os
import re

#set the root directory to the current working directory
root_dir = os.getcwd()

def extract_comments_to_excel(file_path):
    #read the csv file
    df = pd.read_csv(file_path)
    #extract 1000 random comments heading from the dataframe.
    df = df[df["lang"] == "en"]["textCleaned"].sample(n=1000, random_state=1)
    
    #input the comments into a excel file
    df.to_excel("comments.xlsx", index=False)
    print("Excel file created successfully.")
    
def extract_comments_to_csv(file_path):
    #read the csv file
    df = pd.read_csv(file_path)
    #extract 1000 random comments heading from the dataframe.
    df = df[df["lang"] == "en"].sample(n=1000, random_state=1)
    
    #input the comments into a csv file
    df.to_csv("comments.csv", index=False)
    print("CSV file created successfully.")
    
if __name__ == "__main__":
    file_path = os.path.join(root_dir, "preprocessing", "cleaned_youtube_comments.csv")
    extract_comments_to_excel(file_path)
    extract_comments_to_csv(file_path)
    
    
    
    