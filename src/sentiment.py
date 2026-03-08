import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import os
import numpy as np

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class SentimentAnalyzer:
    def __init__(self, model_path="models/sentiment_model.pkl", vectorizer_path="models/vectorizer.pkl"):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.model = None
        self.vectorizer = None
        self.load_models()

    def load_models(self):
        """Load trained sentiment model and vectorizer."""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
        else:
            print(f"Sentiment models not found at {self.model_path}. Please train them.")

    def clean_text(self, text):
        """Preprocess text: clean, tokenize, remove stopwords, lemmatize."""
        if not isinstance(text, str): return ""
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

    def predict_sentiment(self, text):
        """Predict sentiment and return score [-1, 1]."""
        if not self.model or not self.vectorizer:
            # Fallback to dummy sentiment if not trained
            return "Neutral", 0.0, 0.5
        
        cleaned = self.clean_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        # Use predict_proba for confidence if available
        try:
            proba = self.model.predict_proba(vectorized)[0]
            confidence = np.max(proba)
        except:
            confidence = 1.0

        # Map labels to scores
        sentiment_map = {"Positive": 1.0, "Negative": -1.0, "Neutral": 0.0}
        score = sentiment_map.get(prediction, 0.0)
        
        return prediction, score, confidence
