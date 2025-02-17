import pandas as pd
import csv
import re

# Input & Output file names
input_file = "/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/preprocessing/youtube_comments_english.csv"
output_file = "/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/preprocessing/youtube_comments_english_cleaned.csv"

# Function to clean text fields
def clean_text(text):
    if pd.isna(text):  # Handle NaN values
        return ""
    text = str(text)
    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'[^\x20-\x7E]', '', text)  # Remove non-printable characters
    return text

# Read the CSV safely
try:
    df = pd.read_csv(input_file, dtype=str, encoding="utf-8", quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
    
    # Apply text cleaning to all columns
    df = df.applymap(clean_text)
    
    # Remove any completely empty rows
    df.dropna(how='all', inplace=True)

    # Save the cleaned CSV with proper encoding
    df.to_csv(output_file, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    print(f"✅ Fixed CSV saved as {output_file}. Ready for BigQuery upload!")
except Exception as e:
    print(f"❌ Error processing CSV: {e}")
