# Import Libraries
import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')


# Load Dataset
df = pd.read_csv("smart_lms_dataset.csv")

print("✅ Dataset Loaded")
df[["feedback_text", "sentiment"]].head()

# Text Preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)   # remove symbols
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

df["clean_text"] = df["feedback_text"].apply(clean_text)

print("✅ Text Preprocessing Done")

# Encode Target (Sentiment)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["sentiment"] = le.fit_transform(df["sentiment"])

# Train-Test Split
X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data Split Done")

# Convert Text → Numbers (TF-IDF)
vectorizer = TfidfVectorizer(max_features=3000)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("✅ Text Vectorization Done")

# Train Model
model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

print("✅ NLP Model Training Completed")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("🎯 Accuracy:", accuracy)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test with Custom Input (IMP 🔥)
def predict_sentiment(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    pred = model.predict(vector)
    return le.inverse_transform(pred)[0]

# Example
print(predict_sentiment("This course is very helpful and amazing"))
print(predict_sentiment("Worst learning experience ever"))
