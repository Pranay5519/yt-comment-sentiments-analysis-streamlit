import os
import mlflow
from mlflow.tracking import MlflowClient
import joblib
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
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "2", vec_path)
filename = 'yt_comment_model_lightgbm.joblib'
joblib.dump(model, filename)
print(f"Model saved to {filename} using joblib!")