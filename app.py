import os
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv 
load_dotenv()
from utils import (
    fetch_comments,
    preprocess_comment,
    predict_sentiment,
    predict_with_timestamps,
    extract_youtube_video_id
)
from charts import (
    show_sentiment_kpis,
    show_sentiment_pie,
    show_sentiment_trend
)

# ---------------------------------------
# Load model & vectorizer (ONCE)
# ---------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(
        r"C:\Users\prana\Desktop\PROJECTS\yt-comment-streamlit\models\yt_comment_model_lightgbm.joblib"
    )
    vectorizer = joblib.load(
        r"C:\Users\prana\Desktop\PROJECTS\yt-comment-streamlit\models\tfidf_vectorizer.pkl"
    )
    return model, vectorizer

model, vectorizer = load_artifacts()

# ---------------------------------------
# API Key
# ---------------------------------------
yt_api_key = os.getenv("YOUTUBE_API_KEY")
if not yt_api_key:
    st.error("YOUTUBE_API_KEY environment variable is not set")
    st.stop()

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")
st.title("📊 YouTube Comment Sentiment Analysis")

youtube_url = st.text_input("Enter YouTube Video URL")
max_comments = st.slider("Number of comments", 50, 500, 200)

if st.button("Analyze"):
    if not youtube_url:
        st.error("Please enter a YouTube URL")
        st.stop()

    video_id = extract_youtube_video_id(youtube_url)
    if not video_id:
        st.error("Invalid YouTube URL")
        st.stop()

    st.success(f"Video ID extracted: `{video_id}`")

    # -----------------------------------
    # Fetch comments
    # -----------------------------------
    with st.spinner("Fetching comments..."):
        comments = fetch_comments(
            video_id=video_id,
            api_key=yt_api_key,
            max_comments=max_comments
        )

    if not comments:
        st.error("No comments found")
        st.stop()

    st.success(f"Fetched {len(comments)} comments")

    # -----------------------------------
    # Predict sentiment (WITH TIMESTAMPS)
    # -----------------------------------
    with st.spinner("Predicting sentiment..."):
        predictions = predict_with_timestamps(
            comments,
            model,
            vectorizer
        )

    if isinstance(predictions, dict) and "error" in predictions:
        st.error(predictions["error"])
        st.stop()

    # -----------------------------------
    # Display results
    # -----------------------------------
    df = pd.DataFrame(predictions)

    st.subheader("📌 Sentiment KPIs")
    show_sentiment_kpis(df)

    st.subheader("🥧 Sentiment Distribution")
    show_sentiment_pie(df)

    st.subheader("📈 Sentiment Trend Over Time")
    show_sentiment_trend(df)
    
    
    