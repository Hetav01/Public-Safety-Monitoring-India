import json
import os
import requests
import sys, time 
import pandas as pd
from pprint import pprint
from json.decoder import JSONDecodeError
import datetime

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
                    if not words[1].startswith('Downloadi'): 
                        video_list.append(words[1][:-1])

    elif file_type == 'n' : 
        with open(metadata_file) as f:
            for line in f:
                words = line.rstrip("\n").split()
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



#sys.stdout = open('/dev/stdout', 'w')

if __name__ == '__main__':
    """
    Usage : 

    python comment_downloader.py mode(b/n/c) metadata_filepath save_filepath keys_filepath
    
    b : banned mode
    n : normal mode
    """
    print("Hello")
    
    API_URL = 'https://www.googleapis.com/youtube/v3/commentThreads?key={KEY}&textFormat=plainText&part=snippet&videoId={VIDEO_ID}&maxResults=100&pageToken={next_page_token}'
    keys_filepath = sys.argv[4]   #"youtube_data/keys.txt"
    metadata_filepath = sys.argv[2]   #"FoxMetadata.txt"
    save_filepath = sys.argv[3]       #"test"
    mode = sys.argv[1]

    video_list = file_to_vid_list(metadata_filepath, mode)
    KEYS = get_keys(keys_filepath)
    
    print(f"{len(video_list)} videos in list")
    counter = 0 
    nextPgToken = None

    currKeyIdx = 0

    # iterating on the videos - 
    # main loop 
    for video_id in video_list:
        
        #possible_dupe = glob.glob(f"fox_comments/{videoId}")
        if os.path.exists(f"{save_filepath}/{video_id}"):
            print("This video has been iterated over")
            continue 
        else : 
            pass 
        
        count = 0 
        pg = 0  #denotes the page number 
        print('Cur video id: ', video_id)
        print('Cur pg: ', pg)

        nextPageToken = ""
        retryCounter = 20

        
        while True:
            next_url = API_URL
            next_url = API_URL
            
            try : 
                if not nextPageToken: 
                    next_url = API_URL.replace('&pageToken={next_page_token}', '')
                    next_url = next_url.format(KEY=KEYS[currKeyIdx],VIDEO_ID = video_id)

                else :
                    next_url = next_url.format(KEY=KEYS[currKeyIdx],VIDEO_ID=video_id,next_page_token=nextPageToken)

                print('The url: ', next_url)
                
            except IndexError : 
                print("Keys exhausted")
                sys.exit(0)
            
            try:
                req = requests.get(next_url)
                print(req.status_code)
                print('Video id: ', video_id, 'Page:', str(pg))
                if req.status_code == 403 and json.loads(req.text)['error']['errors'][0]['reason'] == 'commentsDisabled':
                    break

                if req.status_code == 404:
                    break

                req.raise_for_status()

                cmts = req.text
                this_pg_payload = json.loads(cmts)
                prettified = json.dumps(this_pg_payload, indent=4)


                if not os.path.exists("{}/{}/".format(save_filepath, video_id)):
                    os.makedirs("{}/{}".format(save_filepath, video_id))
                
                
                path = "{}/{}".format(save_filepath, video_id)
                cur_dest = os.path.join(path, video_id + str(pg) + '.json')

                with open(cur_dest, 'w') as handle:
                    handle.write(prettified)

            

                pg += 1

                if not 'nextPageToken' in this_pg_payload:
                    print(count)    
                    counter += count 
                    break

                nextPageToken = this_pg_payload['nextPageToken']
                print("Next Page Token :",nextPageToken)
                count += 100

            except:
                
                if retryCounter > 0 : 
                    time.sleep(5)
                    retryCounter -= 1 
                    continue
                # advance the keys a bit
                currKeyIdx += 1
                print("advancing keys")
                
                # exit if all the api keys are exhausted
                if currKeyIdx >= len(KEYS):
                    print("Ran out of keys")
                    sys.exit(0)
        