import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Example dataset (Replace with actual data)
data = {
    "text": ["Fake news about elections", "Genuine news about economy", "Clickbait headline", "Verified report on climate"],
    "label": [1, 0, 1, 0]  # 1 = Fake, 0 = Real
}
df = pd.DataFrame(data)

# Train a simple fake news classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('model', MultinomialNB())
])

pipeline.fit(df["text"], df["label"])

# Save the trained model
with open("news_verification_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model saved successfully!")
