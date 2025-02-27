# reddit_fetcher/ml_model.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
from django.conf import settings

class FakeNewsDetector:
    def __init__(self):
        # Create path for model storage
        self.model_dir = os.path.join(settings.BASE_DIR, 'reddit_fetcher', 'ml_models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, 'fake_news_model.pkl')
        self.vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
        
        # Initialize NLP tools
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load or train model
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.load_model()
        else:
            self.train_model()
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and stem
        cleaned_tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(cleaned_tokens)
    
    def train_model(self):
        """Train a fake news detection model"""
        print("Training fake news detection model...")
        
        # For this example, we'll use a simple fake dataset
        # In a real application, you would want to use a proper dataset like Kaggle's Fake News dataset
        
        # Create sample data - in a real scenario, load this from a CSV file
        sample_data = {
            'title': [
                'Scientists discover new species in Amazon',
                'Breaking: COVID-19 vaccine found to be 100% effective',
                'Politician promises to eliminate all taxes',
                'Shocking study reveals coffee cures cancer',
                'New study links vaccines to autism',
                'Government secretly controlling weather',
                'Celebrity announces new movie role',
                'Company reports quarterly earnings',
                'Sports team wins championship',
                'Local community starts recycling program'
            ],
            'text': [
                'Researchers have documented a previously unknown species of frog in the Amazon rainforest.',
                'Phase 3 trials show the new vaccine prevents COVID-19 symptoms in 95% of participants.',
                'During campaign speech, candidate promises to completely eliminate all taxes if elected.',
                'A recent study claims drinking coffee daily eliminates cancer risk by 80%.',
                'Research paper connects childhood vaccines to increased autism rates despite scientific consensus.',
                'Leaked documents show weather manipulation technology being used by government agencies.',
                'The actor will star in upcoming film directed by award-winning director.',
                'Technology company exceeded analyst expectations with 15% growth in second quarter.',
                'The team won their final game of the season, securing the championship title.',
                'Residents partner with local government to implement new waste management system.'
            ],
            'label': [1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # 1 = real, 0 = fake
        }
        
        df = pd.DataFrame(sample_data)
        
        # Preprocess data
        df['processed_text'] = df['title'] + ' ' + df['text']
        df['processed_text'] = df['processed_text'].apply(self.preprocess_text)
        
        # Split data
        X = df['processed_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vectors = self.vectorizer.fit_transform(X_train)
        X_test_vectors = self.vectorizer.transform(X_test)
        
        # Train model (Logistic Regression for simplicity)
        self.model = LogisticRegression()
        self.model.fit(X_train_vectors, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_vectors)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        
        # Save model and vectorizer
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print("Model trained and saved successfully.")
    
    def load_model(self):
        """Load pretrained model and vectorizer"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Loaded pretrained fake news detection model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.train_model()
    
    def predict(self, title, url=None):
        """Predict if news is fake or real"""
        # Preprocess input
        processed_text = self.preprocess_text(title)
        
        # Vectorize
        text_vector = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        probability = self.model.predict_proba(text_vector)[0]
        
        # Generate confidence score
        confidence = probability[1] if prediction == 1 else probability[0]
        
        # Determine credibility level
        if confidence >= 0.8:
            credibility = "High" if prediction == 1 else "Very Low"
        elif confidence >= 0.6:
            credibility = "Medium" if prediction == 1 else "Low"
        else:
            credibility = "Uncertain"
        
        # Return result
        return {
            'is_real': bool(prediction),
            'confidence': float(confidence),
            'credibility': credibility,
            'prediction': "Real News" if prediction == 1 else "Fake News"
        }