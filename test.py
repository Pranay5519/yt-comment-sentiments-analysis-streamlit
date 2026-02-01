from api_helper import (
    fetch_comments_api,
    predict_comments,
    predict_with_timestamps_api,
    generate_chart_api,
    generate_wordcloud_api,
    generate_trend_graph_api,
    extract_youtube_video_id,
)
import os
video_id = "lIo9FcrljDk"
api_key = os.getenv("YOUTUBE_API_KEY") 
comments = fetch_comments_api(
        video_id=video_id,
        api_key=api_key,
        max_comments=500
    )

print("Comments Done")
print(comments)
predictions = predict_comments(comments=comments)
print("Predictions Done")

print(predictions)