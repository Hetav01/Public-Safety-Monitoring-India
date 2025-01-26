import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import emoji
import langdetect
# import google_trans_new
# from google_trans_new import google_translator
# from deep_translator import GoogleTranslator
from langdetect import detect
from translate import Translator


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

#write a function to translate the comments to english
def translate_to_english(comment):
    # translator = GoogleTranslator(source='auto', target='en').translate(text= comment)
    # return translator
    
    # use google_trans_new to translate the comments
    # translator = google_translator()
    # return translator.translate(comment, lang_tgt='en')
    
    #use translate API to translate the comments, but translate only the comments that are not in english keep the rest same.
    if detect_lang(comment) != 'en':
        translator = Translator(to_lang='en')
        return translator.translate(comment)
    return comment    

#translate the comments to english
df['textOriginalEnglish'] = df['textOriginal'].apply(translate_to_english)

# --- Commented out code below is for testing purposes ---

# #translate the comments to english
# df['textOriginalEnglish'] = df['textOriginal'].apply(translate_to_english)

#translate only the comments that are not in english
# print(GoogleTranslator().get_supported_languages(as_dict=True))

#save the translated comments from textOriginalEnglish column in each line of a text file
with open('translated_comments.txt', 'w') as f:
    for comment in df['textOriginalEnglish']:
        f.write(comment + '\n')
        
#update the language in a new column after detecting again
df['langNew'] = df['textOriginalEnglish'].apply(detect_lang)

# save the dataframe to csv
df.to_csv('youtube_comments_english.csv')

print(df.head())
print(df.shape)
print(df['textOriginalEnglish'][84:105])
print(df.info())
print(df['langNew'].value_counts())

