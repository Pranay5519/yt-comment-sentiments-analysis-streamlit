import requests
from typing import List, Dict, Any
from datetime import datetime

# Base URL for the FastAPI server
BASE_URL = "http://localhost:8000"

def fetch_comments(video_id: str, api_key: str, max_comments: int = 100) -> List[Dict]:
    """Fetch comments from YouTube video"""
    url = f"{BASE_URL}/fetch_comments"
    payload = {
        "video_id": video_id,
        "api_key": api_key,
        "max_comments": max_comments
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

def predict_sentiment(comments: List[Dict]) -> List[Dict]:
    """Predict sentiment for comments"""
    url = f"{BASE_URL}/predict"
    payload = {"comments": comments}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

def predict_with_timestamps(comments: List[Dict]) -> List[Dict]:
    """Predict sentiment with timestamps"""
    url = f"{BASE_URL}/predict_with_timestamps"
    payload = {"comments": comments}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

def generate_chart(sentiment_counts: Dict[str, int]) -> bytes:
    """Generate sentiment pie chart"""
    url = f"{BASE_URL}/generate_chart"
    payload = {"sentiment_counts": sentiment_counts}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.content

def generate_wordcloud(comments: List[Dict]) -> bytes:
    """Generate word cloud from comments"""
    url = f"{BASE_URL}/generate_wordcloud"
    payload = {"comments": comments}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.content

def generate_trend_graph(sentiment_data: List[Dict]) -> bytes:
    """Generate sentiment trend graph"""
    url = f"{BASE_URL}/generate_trend_graph"
    payload = {"sentiment_data": sentiment_data}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.content