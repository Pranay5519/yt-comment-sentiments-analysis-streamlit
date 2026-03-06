import streamlit as st
from utils.basic_utils import load_css

# ----------------PAGE CONFIG ----------------
st.set_page_config(
    page_title="YouTube Comments Analyzer",
    page_icon="🎥",
    layout="wide"
)

load_css(r"C:\Users\prana\Desktop\PROJECTS\yt-comment-streamlit\styles\home.css")

# ----------------HEADER ----------------
st.markdown('<div class="main-title">🎥 YouTube Comments Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">End-to-End ML System for Real-Time YouTube Comment Intelligence</div>',
    unsafe_allow_html=True
)
# ----------------API CONFIGURATION ----------------
st.markdown("""
<div class="card">
    <div class="section-header">🔑 API Configuration</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    youtube_key = st.text_input(
        "YouTube API Key",
        type="password",
        value=st.session_state.get("youtube_api_key", "")
    )

with col2:
    google_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=st.session_state.get("google_api_key", "")
    )

save_btn = st.button("Save API Keys")

if save_btn:

    if not youtube_key or not google_key:
        st.error("Please provide both API keys")
    else:
        st.session_state["youtube_api_key"] = youtube_key
        st.session_state["google_api_key"] = google_key
        
        st.success("API Keys saved for this session")
# ----------------PROJECT OVERVIEW ----------------
st.markdown("""
<div class="card">
    <div class="section-header">Project Overview</div>
    <p>
        This project is a production oriented end to end Machine Learning system that analyzes
        real world YouTube comments to understand audience reaction at scale.
    </p>
    <p>
        Given a YouTube video link, the application processes comments and provides structured insights
        about viewer sentiment and discussion themes in real time.
    </p>
</div>
""", unsafe_allow_html=True)


# ----------------ML SENTIMENT MODULE ----------------
st.markdown("""
<div class="card">
    <div class="section-header">Machine Learning Sentiment Analysis</div>
    <ul>
        <li class="feature">Classifies comments into Positive Negative and Neutral sentiments</li>
        <li class="feature">Trained on large scale real world YouTube comment data</li>
        <li class="feature">LightGBM selected after extensive experimentation</li>
        <li class="feature">Achieved 86.8 percent test accuracy</li>
        <li class="feature">Designed for efficient batch and real time inference</li>
    </ul>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="card">
    <div class="section-header">LangGraph Based Topic Classification</div>
    <p>
        The system extends beyond sentiment analysis by identifying discussion topics
        present within video comments using a LangGraph powered LLM workflow.
    </p>
    <ul>
        <li class="feature">Automatically detects major conversation themes</li>
        <li class="feature">Context aware understanding of comment intent</li>
        <li class="feature">Adapts categories based on each video discussion</li>
        <li class="feature">Examples include Feedback Questions Suggestions Criticism and Praise</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# ----------------MLOPS STACK ----------------
st.markdown("""
<div class="card">
    <div class="section-header">⚙️ MLOps & Engineering Stack</div>
    <ul>
        <li class="feature">-DVC for data and model version control</li>
        <li class="feature">-Dagshub for experiment collaboration</li>
        <li class="feature">-MLflow for experiment tracking and model comparison</li>
        <li class="feature">-Modular pipeline design (ingestion → training → evaluation → deployment)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ----------------NAVIGATION HELP ----------------
st.markdown("""
<div class="card">
    <div class="section-header">🧭 How to Explore</div>
    <ul>
        <li class="feature">-Sentiment Dashboard → Overall emotional distribution</li>
        <li class="feature">-Comment Explorer → Browse and filter individual comments</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ----------------FOOTER ----------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Machine Learning, LangGraph, and Streamlit
</div>
""", unsafe_allow_html=True)
