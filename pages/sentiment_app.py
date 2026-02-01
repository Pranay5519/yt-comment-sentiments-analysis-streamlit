import streamlit as st
from api_helper import *
from collections import Counter
import pandas as pd

st.set_page_config(page_title="YouTube Sentiment Analysis", layout="wide")

st.title("🎬 YouTube Comment Sentiment Analysis")

# Sidebar for API configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("YouTube API Key", type="password")
    video_id = st.text_input("Video ID", placeholder="e.g., dQw4w9WgXcQ")
    max_comments = st.slider("Max Comments", 10, 500, 100)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📥 Fetch Comments", "🔮 Predict Sentiment", "📊 Analytics", "📈 Trends"])

# Tab 1: Fetch Comments
with tab1:
    st.header("Fetch YouTube Comments")
    
    if st.button("Fetch Comments", type="primary"):
        if not api_key or not video_id:
            st.error("Please provide API Key and Video ID")
        else:
            with st.spinner("Fetching comments..."):
                try:
                    comments = fetch_comments(video_id, api_key, max_comments)
                    st.session_state['comments'] = comments
                    st.success(f"✅ Fetched {len(comments)} comments!")
                    
                    # Display comments
                    df = pd.DataFrame(comments)
                    st.dataframe(df, width='stretch')
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Tab 2: Predict Sentiment
with tab2:
    st.header("Sentiment Prediction")
    
    if 'comments' not in st.session_state:
        st.warning("⚠️ Please fetch comments first")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Predict Sentiment", type="primary"):
                with st.spinner("Analyzing sentiments..."):
                    try:
                        predictions = predict_sentiment(st.session_state['comments'])
                        st.session_state['predictions'] = predictions
                        st.success("✅ Predictions complete!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("Predict with Timestamps", type="secondary"):
                with st.spinner("Analyzing sentiments with timestamps..."):
                    try:
                        predictions_ts = predict_with_timestamps(st.session_state['comments'])
                        st.session_state['predictions_ts'] = predictions_ts
                        st.success("✅ Predictions with timestamps complete!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Display predictions
        if 'predictions' in st.session_state:
            st.subheader("Prediction Results")
            results = []
            for p in st.session_state['predictions']:
                sentiment_label = {"-1": "😞 Negative", "0": "😐 Neutral", "1": "😊 Positive"}
                results.append({
                    "Comment": p['comment']['text'][:100] + "...",
                    "Sentiment": sentiment_label.get(p['sentiment'], p['sentiment']),
                    "Timestamp": p['comment']['timestamp']
                })
            st.dataframe(pd.DataFrame(results), width='stretch')

# Tab 3: Analytics
with tab3:
    st.header("Sentiment Analytics")
    
    if 'predictions' not in st.session_state:
        st.warning("⚠️ Please run predictions first")
    else:
        # Calculate sentiment counts
        sentiments = [p['sentiment'] for p in st.session_state['predictions']]
        sentiment_counts = Counter(sentiments)
        
        counts_dict = {
            "1": sentiment_counts.get("1", 0),
            "0": sentiment_counts.get("0", 0),
            "-1": sentiment_counts.get("-1", 0)
        }
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("😊 Positive", counts_dict["1"])
        with col2:
            st.metric("😐 Neutral", counts_dict["0"])
        with col3:
            st.metric("😞 Negative", counts_dict["-1"])
        
        # Generate visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            try:
                chart_img = generate_chart(counts_dict)
                st.image(chart_img)
            except Exception as e:
                st.error(f"Chart error: {str(e)}")
        
        with col2:
            st.subheader("Word Cloud")
            try:
                wordcloud_img = generate_wordcloud(st.session_state['comments'])
                st.image(wordcloud_img)
            except Exception as e:
                st.error(f"Word cloud error: {str(e)}")

# Tab 4: Trends
with tab4:
    st.header("Sentiment Trends Over Time")
    
    if 'predictions_ts' not in st.session_state:
        st.warning("⚠️ Please run predictions with timestamps first")
    else:
        try:
            # Prepare data for trend graph
            trend_data = [
                {
                    "timestamp": p['timestamp'],
                    "sentiment": p['sentiment']
                }
                for p in st.session_state['predictions_ts']
            ]
            
            trend_img = generate_trend_graph(trend_data)
            st.image(trend_img, width='stretch')
            
            # Show data table
            with st.expander("View Raw Data"):
                st.dataframe(pd.DataFrame(trend_data), width='stretch')
                
        except Exception as e:
            st.error(f"Trend graph error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and FastAPI")