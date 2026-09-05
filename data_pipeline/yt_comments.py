from googleapiclient.discovery import build
import numpy as np 
import pandas as pd
import os
from dotenv import load_dotenv
from embeddings.essentials import transform_text
from data_pipeline.preprocessingpipeline import pad_word2vec_sequences


load_dotenv()

API_KEY =  os.getenv("YT_API_KEY")

def get_yt_comments(videoid, api = API_KEY):
    # creating youtube resource object
    youtube = build('youtube','v3',developerKey= api)

    # retrieve youtube video results
    video_response=youtube.commentThreads().list(part='snippet',videoId= videoid).execute()

    comments = []
    for item in video_response["items"]:

        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments



def comments_preprocessing(comments):

    processed_comments = transform_text(comments)
    vectors = pad_word2vec_sequences(processed_comments)

    return vectors







