from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

app = Flask(__name__)

# Define the remove_noise function


def remove_noise(text):
    tokens = word_tokenize(text)
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    for token in tokens:
        token = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+', '', token)
        token = lemmatizer.lemmatize(token.lower())
        if len(token) > 1 and token not in stopwords.words('english'):
            clean_tokens.append(token)
    return clean_tokens


# Load the pre-trained models and vectorizers
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
cluster_centers = pickle.load(open("cluster_centers.pkl", "rb"))

# Load the dataset
df = pd.read_csv("job_skills.csv")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_recommendations", methods=["POST"])
def get_recommendations():
    position = request.form["position"]

    # Preprocess the user-provided position
    cleaned_position = " ".join(remove_noise(position))

    # Vectorize the user position using the loaded tfidf_vectorizer
    vectorized_position = tfidf_vectorizer.transform([cleaned_position])

    # Calculate similarity between user position and available positions
    similarities = vectorized_position.dot(cluster_centers.T)

    # Get the indices of top recommendations based on similarity
    top_indices = similarities.argsort(axis=1)[:, ::-1][:, :3]

    # Retrieve the actual position titles for the recommendations
    recommendations = []
    for indices in top_indices:
        recs = []
        for idx in indices:
            recs.append(df.iloc[idx]["Title"])
        recommendations.append(recs)

    return render_template("recommendations.html", recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)
