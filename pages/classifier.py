import streamlit as st
import requests
from utils.api_helper import fetch_comments, extract_youtube_video_id
from utils.basic_utils import load_css

# Load CSS
load_css(r"C:\Users\prana\Desktop\PROJECTS\yt-comment-streamlit\styles\topics.css")

st.set_page_config(page_title="YouTube Comments Topic Analysis", layout="wide")
st.header("🧠 YouTube Comment Topic Classification")

API_BASE_URL = "http://localhost:8000"

# ----------------------------------
# Load API keys from session state
# ----------------------------------

youtube_api_key = st.session_state.get("youtube_api_key")
google_api_key = st.session_state.get("google_api_key")

if not youtube_api_key or not google_api_key:
    st.error("⚠️ Please configure API keys on the Home page first.")
    st.stop()

# ----------------------------------
# Page Inputs (NO SIDEBAR)
# ----------------------------------

col1, col2 = st.columns(2)

with col1:
    youtube_url = st.text_input(
        "Enter YouTube Video URL",
        placeholder="https://youtube.com/watch?v=..."
    )

with col2:
    max_comments = st.slider("Number of comments", 50, 1000, 200)

analyze_btn = st.button("Analyze Topics")

# ----------------------------------
# Main logic
# ----------------------------------

if analyze_btn:

    if not youtube_url:
        st.error("Please enter a YouTube URL")
        st.stop()

    video_id = extract_youtube_video_id(youtube_url)

    if not video_id:
        st.error("Invalid YouTube URL")
        st.stop()

    # ----------------------------------
    # Fetch comments
    # ----------------------------------

    with st.spinner("Fetching comments..."):
        comments = fetch_comments(
            video_id=video_id,
            api_key=youtube_api_key,
            max_comments=max_comments
        )

    if not comments:
        st.warning("No comments found")
        st.stop()

    # ----------------------------------
    # Call FastAPI Topic Endpoint
    # ----------------------------------

    with st.spinner("Discovering topics & classifying comments..."):

        payload = {
            "comments": comments,
            "api_key": google_api_key,
            "model_name": "gemini-2.5-flash",
            "temperature": 0
        }

        response = requests.post(
            f"{API_BASE_URL}/topics",
            json=payload
        )

        if response.status_code != 200:
            st.error("Topic classification API failed")
            st.stop()

        result = response.json()

    topics = result["topics"]
    classified_comments = result["classified_comments"]

    st.success(f"Discovered {len(topics)} topics")

    # ----------------------------------
    # Group comments by topic
    # ----------------------------------

    topic_to_comments = {}

    for item in classified_comments:
        topic = item["topic"]
        comment = item["comment"]

        topic_to_comments.setdefault(topic, []).append(comment)

    # ----------------------------------
    # Render topics
    # ----------------------------------

    st.subheader("📂 Topics & Related Comments")

    for topic in topics:

        comments_for_topic = topic_to_comments.get(topic, [])

        with st.expander(f"🗂 {topic} ({len(comments_for_topic)} comments)", expanded=False):

            if not comments_for_topic:
                st.write("No comments classified under this topic.")

            else:
                for idx, comment in enumerate(comments_for_topic, start=1):
                    st.markdown(f"**{idx}.** {comment}")