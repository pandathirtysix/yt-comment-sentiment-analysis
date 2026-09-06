import os
import pandas as pd
from data_pipeline.yt_comments import get_yt_comments, comments_preprocessing
import numpy as np
import pandas as pd

from urllib.parse import urlparse, parse_qs
from modeltraining.prediction import model_predict
import streamlit as st
from googleapiclient.errors import HttpError

def extract_video_id(link):
    parsed_url = urlparse(link)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None

def comt_with_prediction(comments, prediction):
        #conversion of prediction to human readble
    comment_pred = []
    comment_nd_predict = {"comments": [],
                          "prediction": [],
                          "more detailed" : []}

    for i, j in zip(comments, prediction):
        pred = (
            "negative" if np.argmax(j) == 0
            else "neutral" if np.argmax(j) == 1
            else "positive"
        )

        comment_pred.append(pred)

        comment_nd_predict["comments"].append(i)
        comment_nd_predict["prediction"].append(pred)
        comment_nd_predict["more detailed"].append(j)


    return comment_nd_predict, comment_pred
    

def percentage_of_results(comment_pred):
    total = len(comment_pred)
    
    n = (f"{(comment_pred.count('negative') / total) * 100}%")
    ni = (f"{(comment_pred.count('neutral') / total) * 100}%")
    pos = (f"{(comment_pred.count('positive') / total) * 100}%")

    return n, ni, pos

if __name__ == "__main__":

    st.title("YOUTUBE COMMENT SENTIMENT ANALYISER")

    # vedio id extracion 
    url = st.text_input("Paste the Youtube video Link:", "https://youtu.be/uphThaa97ig?si=V8u6kyv55nzHlksYC")


    button_pred = False
    flag = False

    if url:
        video_id = extract_video_id(url)

        button_pred = st.button("Predict", type="primary")
        if button_pred:
            if video_id:
                try:

                    # st.write("VIDEO ID:", video_id)
                    # comments to vectors (30,100)
                    comments = get_yt_comments(video_id)
                    vectorized_comments = comments_preprocessing(comments)
                    prediction = model_predict(vectorized_comments)
                    flag = True

                except HttpError as e:
                    if "commentsDisabled" in str(e):
                        st.error("Comments are disabled for this video.")
                    else:
                        st.error("Unable to fetch comments for this video")
                    flag = False

    else:
        st.write("paste the url")

    if button_pred and flag:
        cmd_pred, _cmpd = comt_with_prediction(comments, prediction)

        neg, nu, pos = percentage_of_results(_cmpd)

        df = pd.DataFrame(cmd_pred)

        st.session_state["df"] = df
        st.session_state["neg"] = neg
        st.session_state["nu"] = nu
        st.session_state["pos"] = pos


    # Display stored results
    if "df" in st.session_state:
        st.write(f"Negative: {st.session_state['neg']}%")
        st.write(f"Neutral: {st.session_state['nu']}%")
        st.write(f"Positive: {st.session_state['pos']}%")

        button_details = st.button("Detailed View")

        if button_details:
            st.title("Reports:")
            st.dataframe(st.session_state["df"])
                    