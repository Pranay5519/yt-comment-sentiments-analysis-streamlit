import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="YouTube Comment Analyzer",
    page_icon="🎥",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Title */
.main-title {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.5rem;
}

/* Subtitle */
.subtitle {
    font-size: 1.2rem;
    text-align: center;
    color: #d1d5db;
    margin-bottom: 2rem;
}

/* Card style */
.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

/* Section headers */
.section-header {
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Feature list */
.feature {
    font-size: 1.05rem;
    margin-bottom: 8px;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 0.9rem;
    color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🎥 YouTube Comment Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">End-to-End ML & GenAI System for Real-Time YouTube Comment Understanding</div>',
    unsafe_allow_html=True
)

# ---------------- PROJECT OVERVIEW ----------------
st.markdown("""
<div class="card">
    <div class="section-header">📌 Project Overview</div>
    <p>
        This project is an <b>end-to-end Machine Learning and GenAI-powered application</b> that analyzes
        YouTube comments in real time to understand <b>viewer sentiment, intent, and discussion patterns</b>.
    </p>
    <p>
        Users can provide a YouTube video link and instantly gain insights into how audiences are reacting —
        not just emotionally, but <b>contextually</b>.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- ML SENTIMENT MODULE ----------------
st.markdown("""
<div class="card">
    <div class="section-header">🤖 Machine Learning Sentiment Analysis</div>
    <ul>
        <li class="feature">• Classifies comments into <b>Positive, Negative, and Neutral</b> sentiments</li>
        <li class="feature">• Built using real-world YouTube comment data</li>
        <li class="feature">• <b>LightGBM</b> selected after multiple experiments</li>
        <li class="feature">• Achieved <b>86.8% test accuracy</b></li>
        <li class="feature">• Optimized for fast, large-scale inference</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ---------------- GENAI MODULE ----------------
st.markdown("""
<div class="card">
    <div class="section-header">🧠 GenAI-Based Dynamic Comment Classification</div>
    <p>
        Beyond fixed sentiment labels, this application uses <b>Large Language Models (LLMs)</b>
        to perform <b>dynamic, context-aware comment classification</b>.
    </p>
    <ul>
        <li class="feature">• Automatically <b>creates comment categories in real time</b></li>
        <li class="feature">• No predefined labels required</li>
        <li class="feature">• Categories adapt based on the video’s discussion</li>
        <li class="feature">• Examples: <i>Spam, Toxicity, Praise, Feedback, Questions, Suggestions</i></li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ---------------- MLOPS STACK ----------------
st.markdown("""
<div class="card">
    <div class="section-header">⚙️ MLOps & Engineering Stack</div>
    <ul>
        <li class="feature">• <b>DVC</b> for data & model version control</li>
        <li class="feature">• <b>Dagshub</b> for experiment collaboration</li>
        <li class="feature">• <b>MLflow</b> for experiment tracking & model comparison</li>
        <li class="feature">• Modular pipeline design (ingestion → training → evaluation → deployment)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ---------------- NAVIGATION HELP ----------------
st.markdown("""
<div class="card">
    <div class="section-header">🧭 How to Explore</div>
    <ul>
        <li class="feature">• <b>Sentiment Dashboard</b> → Overall emotional distribution</li>
        <li class="feature">• <b>Comment Explorer</b> → Browse & filter individual comments</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Machine Learning, GenAI, and Streamlit
</div>
""", unsafe_allow_html=True)
