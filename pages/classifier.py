import streamlit as st
import requests
from utils.api_helper import fetch_comments, extract_youtube_video_id
from utils.basic_utils import load_css

# ----------------------------------
# Page Setup
# ----------------------------------

load_css(r"C:\Users\prana\Desktop\PROJECTS\yt-comment-streamlit\styles\topics.css")

st.set_page_config(page_title="YouTube Comments Topic Analysis", layout="wide")
st.header("🧠 YouTube Comment Topic Classification")

API_BASE_URL = "http://localhost:8000"

# ----------------------------------
# Load API keys
# ----------------------------------

youtube_api_key = st.session_state.get("youtube_api_key")
google_api_key = st.session_state.get("google_api_key")

if not youtube_api_key or not google_api_key:
    st.error("⚠️ Please configure API keys on the Home page first.")
    st.stop()

# ----------------------------------
# Inputs
# ----------------------------------

col1, col2 = st.columns(2)

with col1:
    youtube_url = st.text_input(
        "Enter YouTube Video URL",
        placeholder="https://youtube.com/watch?v=..."
    )

with col2:
    max_comments = st.slider(
        "Number of comments",
        min_value=50,
        max_value=1000,
        value=200
    )

# ----------------------------------
# Optional Controls
# ----------------------------------

st.subheader("Optional Settings")

user_topics_input = st.text_input(
    "Enter topics manually (comma separated)",
    placeholder="Transformers, Attention Mechanism, RAG Systems"
)

generate_summary = st.radio(
    "Generate topic summaries?",
    ["No", "Yes"],
    horizontal=True
)

generate_summary = generate_summary == "Yes"

analyze_btn = st.button("Analyze Topics")

# ----------------------------------
# Main Logic
# ----------------------------------

if analyze_btn:

    # Convert user topic input → list
    user_topics = []
    if user_topics_input.strip():
        user_topics = [t.strip() for t in user_topics_input.split(",") if t.strip()]

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
    # Call FastAPI
    # ----------------------------------

    with st.spinner("Discovering topics & classifying comments..."):

        payload = {
            "comments": comments,
            "api_key": google_api_key,
            "model_name": "gemini-2.5-flash",
            "temperature": 0,
            "user_topics": user_topics,
            "generate_topic_summary": generate_summary
        }

        try:
            response = requests.post(
                f"{API_BASE_URL}/topics",
                json=payload
            )

            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            st.stop()

        result = response.json()

    # ----------------------------------
    # Extract Results
    # ----------------------------------

    topics = result.get("topics", [])
    classified_comments = result.get("classified_comments", [])
    summary_data = result.get("summary", [])

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
    # Map summaries to topics
    # ----------------------------------

    topic_to_summary = {}

    if summary_data:
        topic_to_summary = {
            item["topic"]: item["summary"]
            for item in summary_data
        }

    # ----------------------------------
    # Render Results
    # ----------------------------------

    st.subheader("📂 Topics & Related Comments")

    for topic in topics:

        comments_for_topic = topic_to_comments.get(topic, [])
        topic_summary = topic_to_summary.get(topic)

        with st.expander(
            f"🗂 {topic} ({len(comments_for_topic)} comments)",
            expanded=False
        ):

            # Show summary if available
            if topic_summary:
                st.markdown("### 📌 Topic Summary")
                st.info(topic_summary)

            if not comments_for_topic:
                st.write("No comments classified under this topic.")
            else:
                st.markdown("### 💬 Comments")
                for idx, comment in enumerate(comments_for_topic, start=1):
                    st.markdown(f"**{idx}.** {comment}")