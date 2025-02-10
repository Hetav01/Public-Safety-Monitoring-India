# So i extracted 1000 comments from the dataset into a csv file on random which is 1000 comments from 500+ different videos.
# What I need to do is to remove these comments from the dataset and keep that dataset in a separate csv file and sort them according to date published.

import pandas as pd
import numpy as np
import os
import re


#read the csv file
df_removed = pd.read_csv("/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/openCoding/openCodingComments.csv")

#read the main csv file from which the comments that are in the openCodingComments.csv file will be removed.
df_main = pd.read_csv("/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/preprocessing/cleaned_youtube_comments.csv")

#print initial number of comments in the main csv file.
print(f"Initial number of comments: {df_main.shape[0]}")
print(df_main.info())

#remove the comments that are in the openCodingComments.csv file from the main csv file.
# if the comment is the same or repeated in the main csv file, remove only one of them.

df_main = df_main[~df_main["etag"].isin(df_removed["etag"])]


#143558 comments should be remaining in the main csv file.
#sort the main csv file according to date published. (don't do it now, do it later, need to check with the professor if the video date matters or the comment date)
# df_main = df_main.sort_values(by=["publishedAt"])    

#count the number of comments in the main csv file.
print(f"Final number of comments: {df_main.shape[0]}")
print(df_main.info())