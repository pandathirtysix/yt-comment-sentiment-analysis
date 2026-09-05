import os
import pandas as pd
from data_pipeline.yt_comments import get_yt_comments, comments_preprocessing
import numpy as np

from urllib.parse import urlparse, parse_qs
from modeltraining.prediction import model_predict

def extract_video_id(link):
    parsed_url = urlparse(link)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None


if __name__ == "__main__":

    # vedio id extracion 
    url =  input("Paste the URL: ")
    video_id = extract_video_id(url) 

    # comments to vectors (30,100)
    comments = get_yt_comments(video_id)
    vectorized_comments = comments_preprocessing(comments)
    prediction = model_predict(vectorized_comments)

    #conversion of prediction to human readble
    comment_pred = []

    for i, j in zip(comments, prediction):
        pred = (
            "negative" if np.argmax(j) == 0
            else "neutral" if np.argmax(j) == 1
            else "postive"
        )

        comment_pred.append(pred)

        print(f"comment: {i}\nprediction: {pred}\n")

    total = len(comment_pred)

    print(f"negative: {(comment_pred.count('negative') / total) * 100}%")
    print(f"neutral: {(comment_pred.count('neutral') / total) * 100}%")
    print(f"positive: {(comment_pred.count('postive') / total) * 100}%")