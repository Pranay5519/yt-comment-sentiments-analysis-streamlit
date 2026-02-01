import os
import streamlit as st
from classifier_agent import topic_graph
from api_helper import fetch_comments, extract_youtube_video_id

st.set_page_config(page_title="YouTube Comments Sentiment Analysis", layout="wide")
st.header("🧠 YouTube Comment Topic Classification")

# ----------------------------------
# Input
# ----------------------------------
youtube_url = st.text_input("Enter YouTube Video URL")

analyze_btn = st.button("Analyze Topics")
max_comments = st.slider("Number of comments", 50, 500, 200)
# ----------------------------------
# Main logic
# ----------------------------------
if analyze_btn:
    if not youtube_url:
        st.error("Please enter a YouTube URL")
        st.stop()

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        st.error("YOUTUBE_API_KEY not set")
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
            api_key=api_key,
            max_comments=500
        )

    if not comments:
        st.warning("No comments found")
        st.stop()

    list_of_comments = [c["text"] for c in comments]

    # ----------------------------------
    # Run topic agent (LIMIT INPUT SIZE)
    # ----------------------------------
    with st.spinner("Discovering topics & classifying comments..."):
        result = topic_graph.invoke({
            "comments": list_of_comments[:20]  # important
        })

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
    # Render expanders
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