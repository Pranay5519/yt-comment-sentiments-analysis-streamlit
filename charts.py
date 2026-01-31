import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------------
# KPI COUNTS
# -----------------------------------
def show_sentiment_kpis(df):
    counts = df["sentiment"].astype(int).value_counts()

    pos = counts.get(1, 0)
    neu = counts.get(0, 0)
    neg = counts.get(-1, 0)

    col1, col2, col3 , col4 = st.columns(4)

    col1.metric("😊 Positive", pos)
    col2.metric("😐 Neutral", neu)
    col3.metric("😡 Negative", neg)
    col4.metric("📊 Total Comments", pos + neu + neg)


# -----------------------------------
# PIE CHART
# -----------------------------------
def show_sentiment_pie(df):
    counts = df["sentiment"].astype(int).value_counts()

    labels = ["Positive", "Neutral", "Negative"]
    sizes = [
        counts.get(1, 0),
        counts.get(0, 0),
        counts.get(-1, 0)
    ]

    if sum(sizes) == 0:
        st.warning("No sentiment data to plot")
        return

    colors = ["#2ecc71", "#95a5a6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(6, 4))  # Reduced from (5, 5)
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors
    )
    ax.axis("equal")

    st.pyplot(fig, use_container_width=False)  # Added use_container_width=False


# -----------------------------------
# SENTIMENT TREND OVER TIME
# -----------------------------------
def show_sentiment_trend(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["sentiment"] = df["sentiment"].astype(int)
    df.set_index("timestamp", inplace=True)

    # Monthly aggregation
    monthly = (
        df.resample("M")["sentiment"]
        .value_counts()
        .unstack(fill_value=0)
    )

    # Ensure all sentiments exist
    for s in [-1, 0, 1]:
        if s not in monthly.columns:
            monthly[s] = 0

    # Convert to percentage
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100
    monthly_pct = monthly_pct[[-1, 0, 1]]

    fig, ax = plt.subplots(figsize=(8, 4))  # Reduced from (10, 5)

    ax.plot(monthly_pct.index, monthly_pct[1], label="Positive", color="green", marker="o")
    ax.plot(monthly_pct.index, monthly_pct[0], label="Neutral", color="gray", marker="o")
    ax.plot(monthly_pct.index, monthly_pct[-1], label="Negative", color="red", marker="o")

    ax.set_title("Sentiment Trend Over Time", fontsize=12)  # Reduced font size
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()  # Better spacing

    st.pyplot(fig, use_container_width=False)  # Added use_container_width=False
    
from wordcloud import WordCloud
from nltk.corpus import stopwords   
def show_wordcloud(comments, preprocess_comment):
    """
    comments: List[str]
    preprocess_comment: function to clean text
    """

    if not comments:
        st.warning("No comments available for word cloud")
        return

    # Preprocess comments
    preprocessed_comments = [
        preprocess_comment(comment) for comment in comments
    ]

    # Combine into single text
    text = " ".join(preprocessed_comments)

    if not text.strip():
        st.warning("Text is empty after preprocessing")
        return

    # Generate word cloud
    wordcloud = WordCloud(
        width=900,
        height=450,
        background_color="black",
        colormap="Blues",
        stopwords=set(stopwords.words("english")),
        collocations=False
    ).generate(text)

    # Display in Streamlit
    st.subheader("☁️ Word Cloud")
    st.image(wordcloud.to_array(), use_container_width=True)