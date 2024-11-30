from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time
from selenium import webdriver
import random
from csv import writer
import pandas as pd
from pyvirtualdisplay import Display
import pickle
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")  # Example option, add more if needed

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)


def scroll_to_end(driver):
    start_time = time.time()
    prev_ht=driver.execute_script("return document.documentElement.scrollHeight")
    i = 0
    sleep_time = 2
    while True:
        i+=1
        print(i)
        if i == 50:
            return driver
        if i % 100 == 0:
            print(i, "scrolls executed")
            current_time = time.time()
            if current_time - start_time > (3600 * 5.5):
                print("5 hours elapsed; breaking at i = ", i)
                break
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(sleep_time + random.uniform(0, 1))
        ht=driver.execute_script("return document.documentElement.scrollHeight")
        if prev_ht == ht:
            time.sleep(2)
            print("Entered same height condition at i = ", i)
            sleep_time += 0
            print("Increased sleep time to ", sleep_time)
            
            if sleep_time >= 20:
                print("Sleep time hit 20s; breaking")
                break
            
        prev_ht = ht
    return driver

def write_to_csv(driver):    
    time_start = time.time()
    out_dict = driver.execute_script("var result = []; " +
    "var all = document.getElementsByTagName('a'); " +
    "for (var i=0, max=all.length; i < max; i++) { " +
    "    if(all[i].getAttribute('id') ==  'video-title')" +                    
    "        result.push([all[i].getAttribute('title'), all[i].getAttribute('href')]); " +
    "} " +
    " return result; ")
    #print(out_dict)
    
    with open('/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/PunePorscheTitles.csv', 'w', newline='', encoding='utf-16', errors ='ignore') as csvfile: #change this line if you want to save the results in the different CSV.
        csv_writer = writer(csvfile)
        csv_writer.writerows(out_dict)
    
    time_end = time.time()
    
    print("written in ", (time_end - time_start) / 60)
    
def apply_date_filter(driver):
    # Wait for the "filters" button to appear and click it
    filters_button = WebDriverWait(driver, 3).until(
        EC.element_to_be_clickable((By.XPATH, "/html/body/ytd-app/div[1]/ytd-page-manager/ytd-search/div[1]/div/ytd-search-header-renderer/div[3]/ytd-button-renderer/yt-button-shape/button/yt-touch-feedback-shape/div"))
    )
    filters_button.click()

    # Wait for the "Upload Date" filter option and click it
    upload_date_filter = WebDriverWait(driver, 3).until(
        EC.element_to_be_clickable((By.XPATH, "/html/body/ytd-app/ytd-popup-container/tp-yt-paper-dialog/ytd-search-filter-options-dialog-renderer/div[2]/ytd-search-filter-group-renderer[1]/ytd-search-filter-renderer[5]/a/div"))
    )
    upload_date_filter.click()
    
    # Wait for the "Relevance" filter option and click it
    # relevance_date_filter = WebDriverWait(driver, 7).until(
    #     EC.element_to_be_clickable((By.XPATH, "/html/body/ytd-app/ytd-popup-container/tp-yt-paper-dialog/ytd-search-filter-options-dialog-renderer/div[2]/ytd-search-filter-group-renderer[5]/ytd-search-filter-renderer[1]/a/div"))
    # )
    # relevance_date_filter.click()
    
    # filters_button = WebDriverWait(driver, 2).until(
    #     EC.element_to_be_clickable((By.XPATH, "/html/body/ytd-app/div[1]/ytd-page-manager/ytd-search/div[1]/div/ytd-search-header-renderer/div[3]/ytd-button-renderer/yt-button-shape/button/yt-touch-feedback-shape/div"))
    # )
    # filters_button.click()
    
    # # Wait for the Upload Date filter option and click it
    # sort_by_filter = WebDriverWait(driver, 2).until(
    #     EC.element_to_be_clickable((By.XPATH, "/html/body/ytd-app/ytd-popup-container/tp-yt-paper-dialog[3]/ytd-search-filter-options-dialog-renderer/div[2]/ytd-search-filter-group-renderer[5]/ytd-search-filter-renderer[1]/a/div"))
    # )
    # sort_by_filter.click()
 

    """
        We're keeping the shorts in the dataset because they are relevant to the topic.
    """
    
# def remove_shorts(fileInput, fileOutput):
    
#     with open(fileInput, "r", newline= '', encoding='utf-16') as infile, open(fileOutput, "w", newline= '', encoding='utf-16') as outfile:
#         reader = csv.reader(infile)
#         writer = csv.writer(outfile)
        
#         for row in reader:
#             if "/shorts" not in ','.join(row):
#                 writer.writerow(row)
    
if __name__ == "__main__": 
    
    # input_file = "/Users/ajaykumarpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/PunePorscheTitles.csv"
    # output_file = "/Users/ajaykumarpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/PunePorscheTitles_filtered.csv"

    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Example option, add more if needed

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
        
    driver.get("https://www.youtube.com/results?search_query=pune+porsche+incident") #change this line if you want to use a new search query.
    
    apply_date_filter(driver)
    
    driver = scroll_to_end(driver)
    write_to_csv(driver)
    
    # remove_shorts(input_file, output_file)
    
    
