import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import emoji
import langdetect
# import google_trans_new
from deep_translator import GoogleTranslator
from langdetect import detect

df = pd.read_csv('/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/eda/final_youtube_comments.csv')

df = df.drop(['Unnamed: 0'], axis=1)

#write a function to demojize the comments and not removing the emojis to retain context.
# demojize only the comments that contain emojis

def demojize_if_emoji(comment):
    comment = str(comment)
    if any(emoji.is_emoji(char) for char in comment):
        return emoji.demojize(comment)
    return comment

#write a function to detect the language of the comments
def detect_lang(comment):
    try:
        return detect(comment)
    except:
        return 'unknown'

#write a function to translate the comments to english
def translate_to_english(comment):
    translator = GoogleTranslator(source='auto', target='en').translate(text= comment)
    return translator

#first demojize the comments
# apply the function to the dataframe
df['textOriginal'] = df['textOriginal'].apply(demojize_if_emoji)

# --- Commented out code below is for testing purposes ---

#check if the emojis are converted to text
# print(df[df["etag"] == "63WlAiuKV0xuDtdhCcWrV1wwYzg"]["textOriginal"].head())


# now detect the language of the comments
df['lang'] = df['textOriginal'].apply(detect_lang)

print(df.head())

# --- Commented out code below is for testing purposes ---

#print how many unknown and english comments are there

# print(df['lang'].value_counts())
# print(df[df["lang"] == "unknown"]["textOriginal"].head())
# print("\n")
# print(df[df["lang"] == "id"]["textOriginal"].head())

#try translating some of the "id" comments to english to check if it works

# print(translate_to_english("Insurance ke nam par scam bhi ho raha hai"))
# print(df[df["etag"] == "63WlAiuKV0xuDtdhCcWrV1wwYzg"]["textOriginal"].head())
# print(df["textOriginal"][84])

#translate the comments to english
df['textOriginalEnglish'] = df['textOriginal'].apply(translate_to_english)

print(df['textOriginalEnglish'][84:105])

#save the dataframe to csv
df.to_csv('youtube_comments_english.csv')


