You first need to create a YouTube API key. 

You can find step-by-step instructions here: https://www.youtube.com/watch?v=jykW3AX8pEE

Before you generate the key, if the Google Cloud has a trial version offer going, enabiling that before generating the key will give your
key 10 times more download quota. Consider enabling it.

Copy the API key and save it in a text file Key.txt

Using DownloadVideoURLsFromSearchQuery.py, you can download video urls for a given YouTube search query. 

In order to use a new search query, change line 66. 
In order to save the CSV to a new location, change line 54. 

In this case, Djokovic.CSV will have entries like

Novak Djokovic deported after losing Australia visa battle - BBC News,/watch?v=_LqLCO5WcL4
Novak Djokovic Is Back In Serbia After Deportation From Australia,/watch?v=xHmyV7uFWRI
Djokovic deported after losing visa appeal,/watch?v=jPcvXB6xrsg
The Deportation of Novak Djokovic: EXPLAINED,/watch?v=73eT31C1TUQ
...

You have to write a Python script to extract the video IDs only and generate a file VideoIDs.txt with entries such as 

_LqLCO5WcL4
xHmyV7uFWRI
jPcvXB6xrsg
73eT31C1TUQ
...

Congratulations! You are now ready to download Video Metadata (information such as video likes, when it is published, number of views) and Comments. 

To obtain the video metadata, first create a folder with name say, DjokovicVideoMetadata 

The pythom command to download video metadata will be

python DownloadVideoMetadata.py c VideoIDs.txt DjokovicVideoMetadata Key.txt 


To obtain the comments, first create a folder with name say, DjokovicComments 

The python command to download comments will be 

python DownloadYouTubeComments c VideoIDs.txt DjokovicComments Key.txt 

Good luck!!