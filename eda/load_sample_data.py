import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

with open("/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/PunePorscheComments/__Gob9QeR0c/__Gob9QeR0c0.json") as sample_file:
    data = json.load(sample_file)


# Extract the 'items' key which contains the comments
comments = data.get('items', [])

# Flatten the JSON structure
def flatten_comment(comment):
    flattened = {
        'kind': comment.get('kind'),
        'etag': comment.get('etag'),
        'id': comment.get('id'),
        'channelId': comment.get('snippet', {}).get('channelId'),
        'videoId': comment.get('snippet', {}).get('videoId'),
        'topLevelComment_kind': comment.get('snippet', {}).get('topLevelComment', {}).get('kind'),
        'topLevelComment_etag': comment.get('snippet', {}).get('topLevelComment', {}).get('etag'),
        'topLevelComment_id': comment.get('snippet', {}).get('topLevelComment', {}).get('id'),
        'textDisplay': comment.get('snippet', {}).get('topLevelComment', {}).get('snippet', {}).get('textDisplay'),
        'textOriginal': comment.get('snippet', {}).get('topLevelComment', {}).get('snippet', {}).get('textOriginal'),
        'authorDisplayName': comment.get('snippet', {}).get('topLevelComment', {}).get('snippet', {}).get('authorDisplayName'),
        'authorProfileImageUrl': comment.get('snippet', {}).get('topLevelComment', {}).get('snippet', {}).get('authorProfileImageUrl'),
        'authorChannelUrl': comment.get('snippet', {}).get('topLevelComment', {}).get('snippet', {}).get('authorChannelUrl'),
        'authorChannelId': comment.get('snippet', {}).get('topLevelComment', {}).get('snippet', {}).get('authorChannelId', {}).get('value'),
        'canRate': comment.get('snippet', {}).get('topLevelComment', {}).get('snippet', {}).get('canRate'),
        'viewerRating': comment.get('snippet', {}).get('topLevelComment', {}).get('snippet', {}).get('viewerRating'),
        'likeCount': comment.get('snippet', {}).get('topLevelComment', {}).get('snippet', {}).get('likeCount'),
        'publishedAt': comment.get('snippet', {}).get('topLevelComment', {}).get('snippet', {}).get('publishedAt'),
        'updatedAt': comment.get('snippet', {}).get('topLevelComment', {}).get('snippet', {}).get('updatedAt'),
        'canReply': comment.get('snippet', {}).get('canReply'),
        'totalReplyCount': comment.get('snippet', {}).get('totalReplyCount'),
        'isPublic': comment.get('snippet', {}).get('isPublic')
    }
    return flattened

# Apply the flattening function
flat_comments = [flatten_comment(comment) for comment in comments]

# Convert to DataFrame
df = pd.DataFrame(flat_comments)

# Display the DataFrame
print(df.head())
