from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Set directories
os.makedirs("models", exist_ok=True)

# Sample labels for training sentiment analysis
data = [
    ("Stock market rally continues as tech earnings beat expectations", "Positive"),
    ("Company reports massive losses, stock price plunges", "Negative"),
    ("Economic growth remains steady amidst market fluctuations", "Neutral"),
    ("Investors bullish on upcoming AI release", "Positive"),
    ("Concerns about inflation lead to market sell-off", "Negative"),
    ("Stock price remains flat after analyst downgrade", "Neutral"),
    ("Profits soar, company dividend increased", "Positive"),
    ("SEC investigation announced, share price down", "Negative"),
    ("Quarterly revenue in line with estimates", "Neutral"),
    ("Tech sector under pressure due to rising rates", "Negative"),
    ("Consumer sentiment reaches record high", "Positive"),
    ("Inventory build-up causes concern for manufacturers", "Neutral"),
]

# Create a DataFrame
df = pd.DataFrame(data, columns=['text', 'sentiment'])

# Define cleaning functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Preprocess the data
df['cleaned'] = df['text'].apply(clean_text)

# Initialize TfidfVectorizer and LogisticRegression
vectorizer = TfidfVectorizer(max_features=5000)
model = LogisticRegression()

# Train the model
X = vectorizer.fit_transform(df['cleaned'])
y = df['sentiment']
model.fit(X, y)

# Save the model and vectorizer
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Sentiment model and vectorizer saved to models/.")
