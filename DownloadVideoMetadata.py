import json
import os
import requests
import sys
import pandas as pd
from pprint import pprint
from json.decoder import JSONDecodeError
import datetime
API_URL_ROOT =  "https://www.googleapis.com/youtube/v3/videos?part=snippet%2Cstatistics%2CcontentDetails&id="





def file_to_vid_list(metadata_file, file_type):
    """
    Converts metadata txt file to a list of videos
    """
    
    video_list = []
    
    if file_type == 'b' :
        with open(metadata_file) as f:
            for line in f:
                words = line.strip().split()
                if words[0] == '[youtube]': 
                    video_list.append(words[1][:-1])

    elif file_type == 'n' : 
        with open(metadata_file) as f:
            for line in f:
                words = line.rstrip("\n").split(',')
                video_list.append(words[1])

    elif file_type == 'c' : 
        with open(metadata_file) as f : 
            for line in f :
                video_list.append(line.rstrip("\n"))

    return video_list


def get_keys(key_filepath): 
    """
    Get keys in a list 
    """

    KEYS = [] 
    with open(key_filepath) as k:
        for line in k:
            KEYS.append(line.rstrip("\n"))
    
    return KEYS

def make_string(videos): 
    video_string = "%2C".join([v for v in videos]) 
    result = API_URL_ROOT + video_string  # + "&key={KEY}"
    return result 

#sys.stdout = open('/dev/stdout', 'w')

if __name__ == '__main__':
    """
    Usage : 

    python comment_downloader.py mode(b/n/c) metadata_filepath save_filepath keys_filepath
    
    b : banned mode
    n : normal mode
    """

    # API_URL_ROOT =  "https://www.googleapis.com/youtube/v3/videos?part=snippet%2Cstatistics&id="

    keys_filepath = sys.argv[4]   #"youtube_data/keys.txt"
    metadata_filepath = sys.argv[2]   #"FoxMetadata.txt"
    save_filepath = sys.argv[3]       #"test"
    mode = sys.argv[1]

    video_list = file_to_vid_list(metadata_filepath, mode)
    KEYS = get_keys(keys_filepath)
    
    print(f"{len(video_list)} videos in list")
    counter = 0 
    # nextPgToken = None

    currKeyIdx = 0

    interval = 50
    current_video = 0

    if not os.path.exists(save_filepath):
        os.makedirs(save_filepath)
        

    while current_video < len(video_list) : 
        
        count = 0 
        print('Cur video id: ', video_list[current_video])
        
        
        try : 

            if current_video + interval <= len(video_list) : 
                temp_videos = video_list[current_video:current_video + interval]

            else : 
                temp_videos = video_list[current_video:]

            current_video += interval       
            s = make_string(temp_videos)
            next_url = s + "&key={KEY}".format(KEY = KEYS[currKeyIdx])

            path = "{}/{}_{}".format(save_filepath, current_video - interval, current_video)
            cur_dest = path + ".json"
            
            if os.path.exists(cur_dest): 
                print(f"indices {current_video - interval} to {current_video} exist")
                continue

            # if not nextPageToken: 
            #     next_url = API_URL.replace('&pageToken={next_page_token}', '')
            #     next_url = next_url.format(KEY=KEYS[currKeyIdx],VIDEO_ID = video_id)

            # else :
            #     next_url = next_url.format(KEY=KEYS[currKeyIdx],VIDEO_ID=video_id,next_page_token=nextPageToken)

            print('The url: ', next_url)
            
        except IndexError : 
            print("Keys exhausted")
            sys.exit(0)
        
        try:
            req = requests.get(next_url)
            print(req.status_code)
            print('Video index: ', current_video)
            if req.status_code == 403 and json.loads(req.text)['error']['errors'][0]['reason'] == 'commentsDisabled':
                break

            if req.status_code == 404:
                break

            req.raise_for_status()

            cmts = req.text
            this_pg_payload = json.loads(cmts)
            prettified = json.dumps(this_pg_payload, indent=4)
            
            # if not os.path.exists("{}/{}/".format(save_filepath, video_id)):
            #     os.makedirs("{}/{}".format(save_filepath, video_id))
            
            

            with open(cur_dest, 'w') as handle:
                handle.write(prettified)



        except:
            
            
            # advance the keys a bit
            currKeyIdx += 1
            print("advancing keys")
            
            # exit if all the api keys are exhausted
            if currKeyIdx >= len(KEYS):
                print("Ran out of keys")
                print(f"Start download from index {current_video - interval}")
                sys.exit(0)
    