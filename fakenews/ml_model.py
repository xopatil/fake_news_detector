import os
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from django.conf import settings

class FakeNewsDetector:
    def __init__(self):
        self.model_dir = os.path.join(settings.BASE_DIR, 'reddit_fetcher', 'ml_models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, 'fake_news_model.pkl')
        self.vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
        
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.load_model()
        else:
            self.train_model()
    
    def preprocess_text(self, text):
        """Preprocess the text by cleaning, tokenizing, and stemming."""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def train_model(self):
        """Train a logistic regression model on a synthetic dataset."""
        print("Training fake news detection model using algorithmic approach...")
        # Synthetic dataset – in production, replace with a real dataset
        data = {
            'title': [
                'Scientists discover new species in Amazon rainforest',
                'Breaking: Miracle cure for diabetes found',
                'Government announces new tax reforms',
                'Celebrity scandal shocks fans worldwide',
                'Unbelievable: Man claims to have time-traveled',
                'Study confirms vaccine efficacy',
                'Fake news: Aliens landed in New York',
                'Research proves coffee boosts productivity',
                'Politician involved in corruption scandal',
                'New study reveals health benefits of meditation'
            ],
            'text': [
                'A team of scientists has discovered a new species in the dense Amazon rainforest.',
                'A miracle cure for diabetes has been discovered that promises to change lives.',
                'The government is set to implement major tax reforms in the coming months.',
                'A major celebrity has been involved in a scandal that has shocked fans globally.',
                'A man claims to have time-traveled, sparking controversy and disbelief.',
                'Recent studies have confirmed the high efficacy of the new vaccine.',
                'Reports of aliens landing in New York have been debunked as fake news.',
                'Research indicates that moderate coffee consumption may boost productivity.',
                'A well-known politician is under investigation for corruption charges.',
                'Meditation has been linked to various health benefits in a new study.'
            ],
            'label': [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  # 1 = Real, 0 = Fake
        }
        df = pd.DataFrame(data)
        df['combined'] = df['title'] + " " + df['text']
        df['processed'] = df['combined'].apply(self.preprocess_text)
        
        X = df['processed']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        self.model = LogisticRegression()
        self.model.fit(X_train_vec, y_train)
        
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load_model(self):
        """Load a previously trained model and vectorizer."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Pre-trained fake news model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.train_model()
    
    def predict(self, title, url=None, score=0, comments=0, author=""):
        """
        Predicts whether the news is fake or real based on the title and adjusts the confidence
        using additional metadata (score, comments, and author).
        """
        processed_title = self.preprocess_text(title)
        vector = self.vectorizer.transform([processed_title])
        prediction = self.model.predict(vector)[0]
        probas = self.model.predict_proba(vector)[0]
        confidence = probas[1] if prediction == 1 else probas[0]
        
        # Additional algorithmic adjustments based on metadata
        adjusted_confidence = confidence
        
        # Low score (upvotes) may indicate lower trust
        if score < 5:
            adjusted_confidence *= 0.8
        
        # Few comments might also lower confidence
        if comments < 3:
            adjusted_confidence *= 0.85
        
        # Apply a penalty if the author is in a blacklist
        blacklisted = {"AutoModerator", "FakeNewsBot"}
        if author in blacklisted:
            adjusted_confidence *= 0.5
        
        credibility = "High" if adjusted_confidence >= 0.8 else "Medium" if adjusted_confidence >= 0.6 else "Low"
        prediction_text = "Real News" if prediction == 1 else "Fake News"
        
        return {
            'is_real': bool(prediction),
            'confidence': float(adjusted_confidence),
            'credibility': credibility,
            'prediction': prediction_text
        }
