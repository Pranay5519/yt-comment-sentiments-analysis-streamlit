
import os
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
import joblib
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "Pranay5519"
    repo_name = "yt-comment-sentiment-analysis"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
    return model, vectorizer
# Initialize the model and vectorizer
vec_path = r"C:\Users\prana\Desktop\PROJECTS\yt-comment-streamlit\models\tfidf_vectorizer.pkl"
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "2", vec_path)  # Update paths and versions as needed

# filename = 'yt_comment_model_lightgbm.joblib'
# joblib.dump(model, filename)
# print(f"Model saved to {filename} using joblib!")

import requests

def fetch_comments(video_id: str, api_key: str, max_comments: int = 500):
    comments = []
    page_token = ""

    try:
        while len(comments) < max_comments:
            url = "https://www.googleapis.com/youtube/v3/commentThreads"
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": 100,
                "pageToken": page_token,
                "key": api_key
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if "items" in data:
                for item in data["items"]:
                    snippet = item["snippet"]["topLevelComment"]["snippet"]

                    comment_text = snippet.get("textOriginal")
                    timestamp = snippet.get("publishedAt")
                    author_id = (
                        snippet.get("authorChannelId", {}).get("value", "Unknown")
                    )

                    comments.append({
                        "text": comment_text,
                        "timestamp": timestamp,
                        "authorId": author_id
                    })

                    if len(comments) >= max_comments:
                        break

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    except requests.exceptions.RequestException as e:
        print("Error fetching comments:", e)

    return comments

def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment
    
    
    import pandas as pd

def predict_sentiment(comments , model, vectorizer):
    if not comments:
        return {"error": "No comments provided"}

    try:
        # 1. Preprocess comments
        preprocessed_comments = [
            preprocess_comment(comment["text"]) for comment in comments
        ]

        # 2. Vectorize comments (sparse matrix)
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # 3. Get expected schema columns from MLflow model
        input_schema = model.metadata.get_input_schema()
        expected_columns = input_schema.input_names()

        # 4. Convert sparse matrix to DataFrame
        feature_names = vectorizer.get_feature_names_out()
        df = pd.DataFrame(
            transformed_comments.toarray(),
            columns=feature_names
        )

        # 5. Add missing expected columns AT ONCE (fixes fragmentation)
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            missing_df = pd.DataFrame(
                0.0,
                index=df.index,
                columns=list(missing_cols)
            )
            df = pd.concat([df, missing_df], axis=1)

        # 6. Reorder columns exactly as model expects
        df = df[expected_columns]

        # 7. Predict
        predictions = model.predict(df).astype(str).tolist()

    except Exception as e:
        return {"error": str(e)}

    # 8. Build response
    return [
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(comments, predictions)
    ]
import pandas as pd

def predict_with_timestamps(comments_data , model, vectorizer):
    if not comments_data:
        return {"error": "No comments provided"}

    try:
        # 1. Extract text and timestamps
        comments = [item["text"] for item in comments_data]
        timestamps = [item["timestamp"] for item in comments_data]

        # 2. Preprocess comments
        preprocessed_comments = [
            preprocess_comment(comment) for comment in comments
        ]

        # 3. Vectorize comments
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # 4. Get expected schema columns from MLflow model
        input_schema = model.metadata.get_input_schema()
        expected_columns = input_schema.input_names()

        # 5. Convert sparse matrix to DataFrame
        feature_names = vectorizer.get_feature_names_out()
        df = pd.DataFrame(
            transformed_comments.toarray(),
            columns=feature_names
        )

        # 6. Add missing columns in ONE operation (fixes warning)
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            missing_df = pd.DataFrame(
                0.0,
                index=df.index,
                columns=list(missing_cols)
            )
            df = pd.concat([df, missing_df], axis=1)

        # 7. Reorder columns exactly as model expects
        df = df[expected_columns]

        # 8. Predict
        predictions = model.predict(df).astype(str).tolist()

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # 9. Build response with timestamps
    return [
        {
            "comment": comment,
            "sentiment": sentiment,
            "timestamp": timestamp
        }
        for comment, sentiment, timestamp in zip(
            comments, predictions, timestamps
        )
    ]


def extract_youtube_video_id(url: str):
    """
    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    """
    pattern = r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None