import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import hashlib
from urllib.parse import urlparse
import joblib
import os

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Set page configuration
st.set_page_config(
    page_title="Reddit Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create a custom CSS style
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B4BFF;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
        border-left: 5px solid #FF4B4B;
    }
    .good-credibility {
        border-left: 5px solid #00CC66;
    }
    .questionable-credibility {
        border-left: 5px solid #FFAA00;
    }
    .poor-credibility {
        border-left: 5px solid #FF4B4B;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin-bottom: 2rem;
    }
    .metric-block {
        text-align: center;
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Reddit API
@st.cache_resource
def initialize_reddit():
    # You should replace these with your Reddit API credentials
    return praw.Reddit(
        client_id="F2AyVPKAxPApH3arBONu9w",
        client_secret="06NmSk2W3V_nd2kllhnzsp1Oq3IRMA",
        user_agent="script:fakenews:1.0 (by u/Amazing-Bite-957)"
    )

# Function to extract domain from URL (already defined in your code)
def extract_domain(url):
    try:
        domain = urlparse(url).netloc
        # Remove 'www.' if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return ""

# Database connection function
def setup_database():
    conn = sqlite3.connect('credibility_data.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS domain_credibility (
        domain TEXT PRIMARY KEY,
        credibility_score REAL,
        last_checked TIMESTAMP,
        verified_count INTEGER,
        is_credible INTEGER,
        is_problematic INTEGER
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS training_data (
        post_id TEXT PRIMARY KEY,
        title TEXT,
        domain TEXT,
        is_self INTEGER,
        score INTEGER,
        upvote_ratio REAL,
        num_comments INTEGER,
        author_age_days REAL,
        author_karma INTEGER,
        sentiment_compound REAL,
        word_count INTEGER,
        avg_word_length REAL,
        title_has_clickbait INTEGER,
        credibility_score REAL,
        is_credible INTEGER,
        feedback INTEGER
    )
    ''')
    
    conn.commit()
    return conn

# Function to check domain credibility (already defined in your code)
def check_domain_credibility(domain):
    conn = setup_database()
    c = conn.cursor()
    
    # Check if we have recent data in our database
    c.execute("SELECT * FROM domain_credibility WHERE domain = ? AND last_checked > datetime('now', '-7 day')", (domain,))
    recent_data = c.fetchone()
    
    if recent_data:
        # Use cached data if it's recent
        domain_data = {
            "domain": recent_data[0],
            "credibility_score": recent_data[1],
            "last_checked": recent_data[2],
            "verified_count": recent_data[3],
            "is_credible": bool(recent_data[4]),
            "is_problematic": bool(recent_data[5])
        }
        conn.close()
        return domain_data
    
    # Base credibility lists
    credible_domains = [
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", 
        "washingtonpost.com", "nytimes.com", "wsj.com", "economist.com",
        "hindustantimes.com", "indianexpress.com", "thehindu.com", "ndtv.com",
        "cnn.com", "theguardian.com", "bloomberg.com", "ft.com", 
        "nature.com", "science.org", "scientificamerican.com"
    ]
    
    problematic_domains = [
        "naturalnews.com", "infowars.com", "breitbart.com", "dailybuzzlive.com",
        "worldnewsdailyreport.com", "empirenews.net", "nationalreport.net",
        "theonion.com", "clickhole.com"  # Satirical sites marked as problematic
    ]
    
    # Initialize credibility data
    domain_data = {
        "domain": domain,
        "credibility_score": 0.5,  # Neutral starting point
        "last_checked": datetime.now().isoformat(),
        "verified_count": 1,
        "is_credible": domain in credible_domains,
        "is_problematic": domain in problematic_domains
    }
    
    # Check external APIs for credibility information
    try:
        # Mock function to simulate an API call
        def mock_fact_check_api(domain):
            # Hash the domain to get a consistent but random-looking score
            hash_val = int(hashlib.md5(domain.encode()).hexdigest(), 16) % 100
            
            # Give higher scores to known credible domains
            if domain in credible_domains:
                base_score = 0.8 + (hash_val % 20) / 100
            # Give lower scores to known problematic domains
            elif domain in problematic_domains:
                base_score = 0.1 + (hash_val % 30) / 100
            # Random but consistent score for unknown domains
            else:
                base_score = 0.3 + (hash_val / 100)
            
            return {
                "credibility": min(1.0, base_score),
                "factual_reporting": "HIGH" if base_score > 0.7 else "MIXED" if base_score > 0.4 else "LOW"
            }
        
        # Get data from our mock API
        result = mock_fact_check_api(domain)
        
        # Update our domain data
        domain_data["credibility_score"] = result["credibility"]
        domain_data["is_credible"] = result["credibility"] > 0.7 or domain in credible_domains
        domain_data["is_problematic"] = result["credibility"] < 0.3 or domain in problematic_domains
        
    except Exception as e:
        st.warning(f"Could not verify domain credibility via API: {str(e)}")
        # Fall back to our base lists if API fails
        domain_data["is_credible"] = domain in credible_domains
        domain_data["is_problematic"] = domain in problematic_domains
        
        # Adjust score based on known lists
        if domain_data["is_credible"]:
            domain_data["credibility_score"] = 0.9
        elif domain_data["is_problematic"]:
            domain_data["credibility_score"] = 0.1
    
    # Store the results in our database
    c.execute('''
    INSERT OR REPLACE INTO domain_credibility 
    (domain, credibility_score, last_checked, verified_count, is_credible, is_problematic)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        domain_data["domain"],
        domain_data["credibility_score"],
        domain_data["last_checked"],
        domain_data["verified_count"],
        int(domain_data["is_credible"]),
        int(domain_data["is_problematic"])
    ))
    
    conn.commit()
    conn.close()
    
    return domain_data

# Function to get text complexity metrics
def get_text_complexity(text):
    if not text:
        return {"word_count": 0, "avg_word_length": 0, "sentence_count": 0}
    
    # Count words
    words = text.split()
    word_count = len(words)
    
    # Calculate average word length
    if word_count > 0:
        avg_word_length = sum(len(word) for word in words) / word_count
    else:
        avg_word_length = 0
        
    # Count sentences (simple approximation)
    sentences = text.split('. ')
    sentence_count = len(sentences)
    
    return {
        "word_count": word_count,
        "avg_word_length": avg_word_length,
        "sentence_count": sentence_count
    }

# Function to check for clickbait patterns in titles
def check_clickbait(title):
    # Common clickbait patterns
    clickbait_patterns = [
        "you won't believe", "shocking", "mind blowing", "jaw dropping",
        "this will make you", "top", "best", "worst", "!!", "???", 
        "secret", "trick", "simple trick", "what happens next", "number",
        "revealed", "that will", "make you", "before you die"
    ]
    
    # Calculate a clickbait score based on pattern matches
    title_lower = title.lower()
    matches = sum(1 for pattern in clickbait_patterns if pattern in title_lower)
    
    # Return 1 if there's likely clickbait, 0 otherwise
    return 1 if matches >= 1 else 0

# Function to fetch Reddit data (modified from your existing code)
def fetch_reddit_data(subreddit_name, time_filter, limit=1000):
    reddit = initialize_reddit()
    conn = setup_database()
    c = conn.cursor()
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        
        # Calculate start and end timestamps based on time_filter
        end_time = datetime.utcnow()
        if time_filter == "hour":
            start_time = end_time - timedelta(hours=1)
        elif time_filter == "day":
            start_time = end_time - timedelta(days=1)
        elif time_filter == "week":
            start_time = end_time - timedelta(weeks=1)
        elif time_filter == "month":
            start_time = end_time - timedelta(days=30)
        elif time_filter == "year":
            start_time = end_time - timedelta(days=365)
        else:  # All time
            start_time = datetime.utcfromtimestamp(0)
        
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        # Collection for returning the data
        all_posts_data = []
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        posts_processed = 0
        
        # Use pushshift.io API to get post IDs for the date range
        pushshift_url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit_name}&after={start_timestamp}&before={end_timestamp}&size={limit}"
        
        try:
            import requests
            response = requests.get(pushshift_url)
            if response.status_code == 200:
                post_ids = [post['id'] for post in response.json()['data']]
                total_posts = len(post_ids)
                
                # Fetch full post data using PRAW
                for i, post_id in enumerate(post_ids):
                    try:
                        submission = reddit.submission(id=post_id)
                        posts_processed += 1
                        
                        # Update progress
                        progress = min(posts_processed / total_posts, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing post {posts_processed}/{total_posts}: {submission.title[:50]}...")
                        
                        # Get author data 
                        if submission.author:
                            author_name = submission.author.name
                            try:
                                author_created_utc = submission.author.created_utc
                                author_age_days = (time.time() - author_created_utc) / (60 * 60 * 24)
                                author_comment_karma = submission.author.comment_karma
                                author_link_karma = submission.author.link_karma
                                author_has_verified_email = submission.author.has_verified_email
                            except:
                                # Fallback if author data can't be retrieved
                                author_created_utc = time.time() - 30 * 86400  # Default 30 days
                                author_age_days = 30
                                author_comment_karma = 100
                                author_link_karma = 50
                                author_has_verified_email = False
                        else:
                            author_name = "[deleted]"
                            author_created_utc = time.time() - 30 * 86400
                            author_age_days = 30
                            author_comment_karma = 100
                            author_link_karma = 50
                            author_has_verified_email = False
                        
                        # Extract domain and check credibility in real-time
                        domain = extract_domain(submission.url)
                        domain_credibility = check_domain_credibility(domain)
                        
                       # Full post data (for processing, not all will be inserted)
                        post_data = {
                            "id": submission.id,
                            "title": submission.title,
                            "url": submission.url,
                            "domain": domain,
                            "is_self": submission.is_self,
                            "selftext": submission.selftext,
                            "score": submission.score,
                            "upvote_ratio": submission.upvote_ratio,
                            "num_comments": submission.num_comments,
                            "created_utc": submission.created_utc,
                            "post_age_days": (time.time() - submission.created_utc) / (60 * 60 * 24),
                            "author": author_name,
                            "author_age_days": author_age_days,
                            "author_comment_karma": author_comment_karma,
                            "author_link_karma": author_link_karma,
                            "author_has_verified_email": author_has_verified_email,
                            "credibility_score": domain_credibility.get("credibility_score", 0.5),
                            "domain_credibility_score": domain_credibility["credibility_score"],
                            "domain_is_credible": int(domain_credibility["is_credible"]),
                            "domain_is_problematic": int(domain_credibility["is_problematic"])
                        }

                        # Add author_karma after the dictionary is created
                        post_data["author_karma"] = author_comment_karma + author_link_karma
                        
                        # Analyze sentiment from title and selftext
                        combined_text = submission.title
                        if submission.selftext:
                            combined_text += " " + submission.selftext
                            
                        sentiment = sid.polarity_scores(combined_text)
                        post_data["sentiment_neg"] = sentiment["neg"]
                        post_data["sentiment_neu"] = sentiment["neu"]
                        post_data["sentiment_pos"] = sentiment["pos"]
                        post_data["sentiment_compound"] = sentiment["compound"]
                        
                        # Text complexity
                        complexity = get_text_complexity(combined_text)
                        post_data["word_count"] = complexity["word_count"]
                        post_data["avg_word_length"] = complexity["avg_word_length"]
                        post_data["sentence_count"] = complexity["sentence_count"]
                        
                        # Check for clickbait
                        post_data["title_has_clickbait"] = check_clickbait(submission.title)
                        
                        # Get top-level comments for sentiment analysis
                        submission.comments.replace_more(limit=0)  # Only get readily available comments
                        comment_sentiments = []
                        for comment in list(submission.comments)[:5]:  # Get top 5 comments
                            if comment.body:
                                comment_sentiment = sid.polarity_scores(comment.body)["compound"]
                                comment_sentiments.append(comment_sentiment)
                        
                        if comment_sentiments:
                            post_data["avg_comment_sentiment"] = np.mean(comment_sentiments)
                            post_data["comment_sentiment_variance"] = np.var(comment_sentiments)
                        else:
                            post_data["avg_comment_sentiment"] = 0
                            post_data["comment_sentiment_variance"] = 0
                        
                        # Prepare the specific data for database insertion
                        data = {
                            "post_id": post_data["id"],
                            "title": post_data["title"],
                            "domain": post_data["domain"],
                            "is_self": int(post_data["is_self"]),
                            "score": post_data["score"],
                            "upvote_ratio": post_data["upvote_ratio"],
                            "num_comments": post_data["num_comments"],
                            "author_age_days": post_data["author_age_days"],
                            "author_karma": post_data.get("author_comment_karma", 0) + post_data.get("author_link_karma", 0),
                            "sentiment_compound": post_data["sentiment_compound"],
                            "word_count": post_data["word_count"],
                            "avg_word_length": post_data["avg_word_length"],
                            "title_has_clickbait": post_data.get("title_has_clickbait", 0),
                            "credibility_score": domain_credibility.get("credibility_score", 0.5),
                            "is_credible": domain_credibility.get("is_credible", 0),
                            "feedback": None  # No user feedback at this stage
                        }
                        
                        # Insert data into the database immediately
                        columns = ", ".join(data.keys())
                        placeholders = ", ".join(["?"] * len(data))
                        
                        query = f'''
                        INSERT OR REPLACE INTO training_data
                        ({columns})
                        VALUES ({placeholders})
                        '''
                        
                        try:
                            # Execute the query with values
                            c.execute(query, list(data.values()))
                            conn.commit()
                            
                            # Also store the full post data if needed for return
                            all_posts_data.append(post_data)
                            
                        except sqlite3.Error as e:
                            st.error(f"Database error for post {submission.id}: {str(e)}")
                            continue
                            
                    except Exception as e:
                        st.warning(f"Skipped post {post_id}: {str(e)}")
                        continue
            else:
                st.error("Failed to fetch data from Pushshift API")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error accessing Pushshift API: {str(e)}")
            return pd.DataFrame()
        
        # Clear the progress indicators
        progress_bar.empty()
        status_text.empty()
                
        return pd.DataFrame(all_posts_data)
        
    except Exception as e:
        st.error(f"Error fetching data from r/{subreddit_name}: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

# Function to load training data from the database
def load_training_data():
    conn = setup_database()
    # Load all labeled training data
    query = "SELECT * FROM training_data WHERE feedback IS NOT NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Function to generate synthetic training data if real data is insufficient
def generate_synthetic_training_data(min_samples=100):
    conn = setup_database()
    
    # First, try to get real training data
    real_data = pd.read_sql_query("SELECT * FROM training_data", conn)
    
    # If we don't have enough real data, generate synthetic data
    if len(real_data) < min_samples:
        st.info(f"Not enough real training data ({len(real_data)} samples). Generating synthetic data...")
        
        # Number of synthetic samples to generate
        n_samples = min_samples - len(real_data)
        
        # Create lists of credible and problematic domains
        credible_domains = [
            "reuters.com", "apnews.com", "bbc.com", "npr.org", 
            "washingtonpost.com", "nytimes.com", "wsj.com"
        ]
        problematic_domains = [
            "naturalnews.com", "infowars.com", "breitbart.com", 
            "worldnewsdailyreport.com", "empirenews.net"
        ]
        
        # Generate synthetic data
        synthetic_data = []
        for i in range(n_samples):
            # Decide if this will be credible or not
            is_credible = np.random.choice([0, 1], p=[0.4, 0.6])
            
            if is_credible:
                domain = np.random.choice(credible_domains)
                credibility_score = np.random.uniform(0.7, 1.0)
                upvote_ratio = np.random.uniform(0.7, 0.95)
                score = np.random.randint(10, 5000)
                sentiment_compound = np.random.uniform(0.0, 0.8)
                title_has_clickbait = np.random.choice([0, 1], p=[0.9, 0.1])
                author_age_days = np.random.uniform(100, 3000)
                author_karma = np.random.randint(1000, 100000)
            else:
                domain = np.random.choice(problematic_domains)
                credibility_score = np.random.uniform(0.0, 0.3)
                upvote_ratio = np.random.uniform(0.5, 0.8)
                score = np.random.randint(0, 1000)
                sentiment_compound = np.random.uniform(-0.8, 0.2)
                title_has_clickbait = np.random.choice([0, 1], p=[0.3, 0.7])
                author_age_days = np.random.uniform(1, 500)
                author_karma = np.random.randint(0, 5000)
            
            # Common fields
            post_id = f"synthetic_{i}"
            title = f"Synthetic Title {i}"
            is_self = np.random.choice([0, 1])
            num_comments = np.random.randint(0, 500)
            word_count = np.random.randint(50, 500)
            avg_word_length = np.random.uniform(4.0, 7.0)
            feedback = 1 if is_credible else 0  # 1 for credible, 0 for fake news
            
            # Create a synthetic sample
            sample = {
                "post_id": post_id,
                "title": title,
                "domain": domain,
                "is_self": is_self,
                "score": score,
                "upvote_ratio": upvote_ratio,
                "num_comments": num_comments,
                "author_age_days": author_age_days,
                "author_karma": author_karma,
                "sentiment_compound": sentiment_compound,
                "word_count": word_count,
                "avg_word_length": avg_word_length,
                "title_has_clickbait": title_has_clickbait,
                "credibility_score": credibility_score,
                "is_credible": is_credible,
                "feedback": feedback
            }
            
            synthetic_data.append(sample)
        
        # Create a DataFrame from the synthetic data
        synthetic_df = pd.DataFrame(synthetic_data)
        
        # Insert synthetic data into the database
        for _, row in synthetic_df.iterrows():
            columns = ", ".join(row.index)
            placeholders = ", ".join(["?"] * len(row))
            query = f"INSERT OR REPLACE INTO training_data ({columns}) VALUES ({placeholders})"
            conn.execute(query, list(row.values))
        
        conn.commit()
        
        # Combine real and synthetic data
        training_data = pd.concat([real_data, synthetic_df], ignore_index=True)
    else:
        training_data = real_data
    
    conn.close()
    return training_data

# Function to train the Random Forest model
def train_model(X, y):
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        # Perform cross-validation
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Train the final model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Save model and scaler
        model_info = {
            'model': model,
            'scaler': scaler,
            'features': list(X.columns),
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'trained_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        joblib.dump(model_info, "fake_news_model_info.joblib")
        
        return model, scaler, accuracy, report, cm, X_test_scaled, y_test, y_pred, cv_scores
        
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None, None, 0, {}, None, None, None, None, []

# Function to predict credibility for new posts
def predict_credibility(model, scaler, new_data):
    try:
        # Get features from training
        features = [
            'is_self', 'score', 'upvote_ratio', 'num_comments', 'author_age_days',
            'author_karma', 'sentiment_compound', 'word_count', 'avg_word_length',
            'title_has_clickbait', 'credibility_score'
        ]
        
        # Prepare features
        X_new = new_data[features].copy()
        
        # Handle missing values
        X_new.fillna(0, inplace=True)
        
        # Scale the features
        X_new_scaled = scaler.transform(X_new)
        
        # Make predictions
        predictions = model.predict(X_new_scaled)
        probabilities = model.predict_proba(X_new_scaled)
        
        # Create a copy of the input DataFrame to avoid modifying it directly
        result_data = new_data.copy()
        
        # Add prediction results
        result_data['is_fake_news'] = predictions
        result_data['probability_fake'] = [prob[0] for prob in probabilities]
        result_data['probability_real'] = [prob[1] for prob in probabilities]
        
        return result_data
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        # Return DataFrame with default prediction values if error occurs
        new_data['is_fake_news'] = 0
        new_data['probability_fake'] = 0.0
        new_data['probability_real'] = 1.0
        return new_data

# Function to update feedback in the database
def update_feedback(post_id, feedback_value):
    conn = setup_database()
    c = conn.cursor()
    
    # Update the feedback value
    c.execute("UPDATE training_data SET feedback = ? WHERE post_id = ?", (feedback_value, post_id))
    conn.commit()
    conn.close()
    
    return True

# Function to get feature importances
def get_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    feature_importance_data = []
    for i in indices:
        feature_importance_data.append({
            'feature': feature_names[i],
            'importance': importances[i]
        })
    
    return pd.DataFrame(feature_importance_data)

# Main function for the Streamlit app
def main():
    # Display header
    st.markdown("<h1 class='main-header'>Reddit Fake News Detector</h1>", unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a mode", 
                                     ["Home", "Fetch Reddit Data", "Train Model", "Analyze Live Posts", "Model Performance"])
    
    # Check if the model has been trained
    model_path = "random_forest_model.joblib"
    scaler_path = "scaler.joblib"
    model_exists = os.path.exists(model_path) and os.path.exists(scaler_path)
    
    # Home page
    if app_mode == "Home":
        st.markdown("""
        ## Welcome to the Reddit Fake News Detector
        
        This application helps you identify potentially fake news on Reddit using machine learning.
        
        ### How it works:
        1. **Fetch Reddit Data**: Collect posts from any subreddit to analyze
        2. **Train Model**: Train a Random Forest classifier using existing data
        3. **Analyze Live Posts**: Make real-time predictions on the credibility of posts
        4. **Model Performance**: View metrics on how well the model is performing
        
        ### Key Features:
        - Domain credibility checking
        - Sentiment analysis
        - Text complexity metrics
        - Clickbait detection
        - User feedback integration
        
        Get started by selecting a mode from the sidebar!
        """)
        
        # Display database stats
        st.markdown("<h2 class='sub-header'>Database Statistics</h2>", unsafe_allow_html=True)
        
        conn = setup_database()
        c = conn.cursor()
        
        # Count posts
        c.execute("SELECT COUNT(*) FROM training_data")
        post_count = c.fetchone()[0]
        
        # Count domains
        c.execute("SELECT COUNT(*) FROM domain_credibility")
        domain_count = c.fetchone()[0]
        
        # Count posts with feedback
        c.execute("SELECT COUNT(*) FROM training_data WHERE feedback IS NOT NULL")
        feedback_count = c.fetchone()[0]
        
        # Close connection
        conn.close()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Posts", post_count)
            
        with col2:
            st.metric("Domains Tracked", domain_count)
            
        with col3:
            st.metric("User Feedback", feedback_count)
        
        # Check if the model exists
        if model_exists:
            st.success("A trained model is available for predictions!")
        else:
            st.warning("No trained model found. Go to the 'Train Model' tab to create one.")
    
    # Fetch Reddit Data page
    elif app_mode == "Fetch Reddit Data":
        st.markdown("<h2 class='sub-header'>Fetch Reddit Data</h2>", unsafe_allow_html=True)
        
        # Input for subreddit and time filter
        subreddit_name = st.text_input("Enter subreddit name (without r/)", "news")
        time_filter = st.selectbox("Select time filter", ["hour", "day", "week", "month", "year", "all"])
        limit = st.slider("Number of posts to fetch", min_value=10, max_value=500, value=50)
        
        if st.button("Fetch Data"):
            with st.spinner("Fetching data from Reddit..."):
                data = fetch_reddit_data(subreddit_name, time_filter, limit)
                
                if not data.empty:
                    st.success(f"Successfully fetched {len(data)} posts from r/{subreddit_name}")
                    
                    # Display sample of the fetched data
                    st.markdown("<h3 class='sub-header'>Sample of Fetched Posts</h3>", unsafe_allow_html=True)
                    st.dataframe(data[['title', 'domain', 'score', 'upvote_ratio', 'num_comments', 'domain_credibility_score']].head(10))
                    
                    # Show domains and their credibility
                    st.markdown("<h3 class='sub-header'>Domain Credibility Analysis</h3>", unsafe_allow_html=True)
                    domain_data = data[['domain', 'domain_credibility_score', 'domain_is_credible', 'domain_is_problematic']].drop_duplicates()
                    
                    # Convert boolean columns for display
                    domain_data['domain_is_credible'] = domain_data['domain_is_credible'].apply(lambda x: 'Yes' if x else 'No')
                    domain_data['domain_is_problematic'] = domain_data['domain_is_problematic'].apply(lambda x: 'Yes' if x else 'No')
                    
                    # Rename columns for display
                    domain_data = domain_data.rename(columns={
                        'domain': 'Domain',
                        'domain_credibility_score': 'Credibility Score',
                        'domain_is_credible': 'Is Credible',
                        'domain_is_problematic': 'Is Problematic'
                    })
                    
                    st.dataframe(domain_data)
                    
                    # Show sentiment distribution
                    st.markdown("<h3 class='sub-header'>Sentiment Analysis</h3>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data['sentiment_compound'], bins=20, kde=True, ax=ax)
                    ax.set_xlabel('Sentiment Compound Score')
                    ax.set_ylabel('Count')
                    ax.set_title('Distribution of Sentiment in Posts')
                    st.pyplot(fig)
                    
                    # Show clickbait analysis
                    st.markdown("<h3 class='sub-header'>Clickbait Analysis</h3>", unsafe_allow_html=True)
                    clickbait_count = data['title_has_clickbait'].sum()
                    clickbait_percentage = (clickbait_count / len(data)) * 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Posts with Clickbait Patterns", f"{clickbait_count} / {len(data)}")
                        
                    with col2:
                        st.metric("Percentage Clickbait", f"{clickbait_percentage:.1f}%")
                        
                    # Show a few examples of clickbait titles if any
                    if clickbait_count > 0:
                        st.markdown("#### Examples of Potential Clickbait Titles:")
                        clickbait_examples = data[data['title_has_clickbait'] == 1][['title', 'score']].head(5)
                        for _, row in clickbait_examples.iterrows():
                            st.markdown(f"- {row['title']} (Score: {row['score']})")
                else:
                    st.error(f"Failed to fetch data from r/{subreddit_name}. Please try a different subreddit or time filter.")
    
    # Train Model page
    elif app_mode == "Train Model":
        st.markdown("<h2 class='sub-header'>Train Fake News Detection Model</h2>", unsafe_allow_html=True)
        
        # Check for existing training data
        conn = setup_database()
        query = """
        SELECT is_self, score, upvote_ratio, num_comments, author_age_days,
               author_karma, sentiment_compound, word_count, avg_word_length,
               title_has_clickbait, credibility_score, is_credible
        FROM training_data
        WHERE is_credible IS NOT NULL
        """
        training_data = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(training_data) < 100:  # Not enough data
            st.warning("Not enough training data. Fetching live data from r/news...")
            with st.spinner("Fetching data from Reddit..."):
                # Fetch live data from r/news
                new_data = fetch_reddit_data("news", "day", limit=200)
                if not new_data.empty:
                    st.success(f"Successfully fetched {len(new_data)} posts from r/news")
                    training_data = new_data[['is_self', 'score', 'upvote_ratio', 'num_comments', 
                                            'author_age_days', 'author_karma', 'sentiment_compound',
                                            'word_count', 'avg_word_length', 'title_has_clickbait',
                                            'credibility_score', 'domain_is_credible']]
                    training_data = training_data.rename(columns={'domain_is_credible': 'is_credible'})
                else:
                    st.error("Failed to fetch data. Please try again.")
                    return
        
        # Display dataset info
        st.write(f"Training dataset size: {len(training_data)} posts")
        
        # Features for training
        features = [
            'is_self', 'score', 'upvote_ratio', 'num_comments', 'author_age_days',
            'author_karma', 'sentiment_compound', 'word_count', 'avg_word_length',
            'title_has_clickbait', 'credibility_score'
        ]
        
        # Train button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Prepare features and target
                    X = training_data[features].copy()
                    y = training_data['is_credible'].astype(int)
                    
                    # Handle missing values
                    X = X.fillna(0)
                    
                    # Verify data
                    if X.isnull().any().any():
                        st.error("Data contains null values after cleaning. Please check the data.")
                        return
                    
                    if len(X) != len(y):
                        st.error("Feature and target dimensions don't match.")
                        return
                    
                    # Train model
                    st.info("Training Random Forest model...")
                    
                    # Initialize model with basic parameters
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        class_weight='balanced',
                        random_state=42
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Train model
                    model.fit(X_scaled, y)
                    
                    # Calculate accuracy
                    y_pred = model.predict(X_scaled)
                    accuracy = accuracy_score(y, y_pred)
                    
                    # Save model and scaler
                    st.info("Saving model...")
                    joblib.dump(model, "random_forest_model.joblib")
                    joblib.dump(scaler, "scaler.joblib")
                    
                    # Show results
                    st.success(f"Model trained successfully with accuracy: {accuracy:.4f}")
                    
                    # Show class distribution
                    st.write("Class Distribution:")
                    st.write(pd.Series(y).value_counts().to_dict())
                    
                    # Show feature importance
                    if hasattr(model, 'feature_importances_'):
                        importances = pd.DataFrame({
                            'feature': features,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        st.write("Top 5 Most Important Features:")
                        st.dataframe(importances.head())
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    st.error("Please try again or check the data quality.")
                    st.write("Data sample for debugging:")
                    st.write(X.head())
                    st.write("Target sample:")
                    st.write(y.head())

    # Analyze Live Posts page
    elif app_mode == "Analyze Live Posts":
        st.markdown("<h2 class='sub-header'>Analyze Live Reddit Posts</h2>", unsafe_allow_html=True)
        
        # Check if model exists
        if not model_exists:
            st.warning("No trained model found. Please go to the 'Train Model' tab to train a model first.")
            
            # Option to generate and train on synthetic data
            if st.button("Quick Start: Generate Data & Train Model"):
                with st.spinner("Generating synthetic data and training model..."):
                    # Generate synthetic data
                    synthetic_data = generate_synthetic_training_data(min_samples=200)
                    
                    # Prepare features and target
                    features = [
                        'is_self', 'score', 'upvote_ratio', 'num_comments', 'author_age_days',
                        'author_karma', 'sentiment_compound', 'word_count', 'avg_word_length',
                        'title_has_clickbait', 'credibility_score'
                    ]
                    
                    # Filter rows with feedback
                    labeled_data = synthetic_data[synthetic_data['feedback'].notna()]
                    
                    X = labeled_data[features]
                    y = labeled_data['feedback']
                    
                    # Handle missing values
                    X.fillna(0, inplace=True)
                    
                    # Train and save the model
                    model, scaler, accuracy, _, _, _, _, _ = train_model(X, y)
                    joblib.dump(model, "random_forest_model.joblib")
                    joblib.dump(scaler, "scaler.joblib")
                    
                    st.success(f"Model trained successfully with accuracy: {accuracy:.4f}")
                    st.experimental_rerun()  # Rerun to update the interface
        else:
            # Load the trained model
            model = joblib.load("random_forest_model.joblib")
            scaler = joblib.load("scaler.joblib")
            
            # Create tabs for bulk analysis and single post analysis
            tab1, tab2 = st.tabs(["Analyze Multiple Posts", "Analyze Single Post"])
            
            with tab1:
                # Original bulk analysis code
                subreddit_name = st.text_input("Enter subreddit name (without r/)", "news")
                time_filter = st.selectbox("Select time filter", ["hour", "day", "week"])
                limit = st.slider("Number of posts to analyze", min_value=5, max_value=100, value=20)
                
                if st.button("Analyze Subreddit"):
                    with st.spinner("Fetching and analyzing posts..."):
                        # Fetch new data
                        new_data = fetch_reddit_data(subreddit_name, time_filter, limit)
                        
                        if not new_data.empty:
                            # Make predictions
                            result_data = predict_credibility(model, scaler, new_data)
                            
                            # Display overall stats
                            st.markdown("<h3 class='sub-header'>Analysis Results</h3>", unsafe_allow_html=True)
                            
                            fake_count = result_data['is_fake_news'].sum()
                            real_count = len(result_data) - fake_count
                            fake_percentage = (fake_count / len(result_data)) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Posts", len(result_data))
                            
                            with col2:
                                st.metric("Potentially Fake", f"{fake_count} ({fake_percentage:.1f}%)")
                            
                            with col3:
                                st.metric("Likely Credible", real_count)
                            
                            # Visualization of results
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(
                                data=result_data, 
                                x='probability_real', 
                                hue='is_fake_news',
                                bins=20, 
                                kde=True,
                                ax=ax
                            )
                            ax.set_xlabel('Probability of being credible')
                            ax.set_ylabel('Count')
                            ax.set_title('Distribution of Credibility Scores')
                            ax.legend(['Fake News', 'Credible'])
                            st.pyplot(fig)
                            
                            # Display individual posts with predictions
                            st.markdown("<h3 class='sub-header'>Post Analysis</h3>", unsafe_allow_html=True)
                            
                            # Sort by probability of being fake (ascending)
                            sorted_data = result_data.sort_values('probability_real')
                            
                            for _, post in sorted_data.iterrows():
                                # Determine card class based on credibility
                                card_class = "poor-credibility" if post['probability_real'] < 0.3 else (
                                    "questionable-credibility" if post['probability_real'] < 0.7 else "good-credibility"
                                )
                                
                                # Display the post card
                                st.markdown(f"<div class='card {card_class}'>", unsafe_allow_html=True)
                                
                                # Post title and credibility score
                                st.markdown(f"**Title**: {post['title']}")
                                st.markdown(f"**Domain**: {post['domain']} (Credibility Score: {post['domain_credibility_score']:.2f})")
                                
                                # Display prediction results
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown(f"**Prediction**: {'❌ Potentially Fake' if post['is_fake_news'] else '✅ Likely Credible'}")
                                
                                with col2:
                                    st.markdown(f"**Confidence**: {max(post['probability_fake'], post['probability_real']):.2f}")  # Fixed format
                                
                                with col3:
                                    # Feedback buttons
                                    post_id = post['id']
                                    col3_1, col3_2 = st.columns(2)
                                    
                                    with col3_1:
                                        if st.button("👎 Fake", key=f"fake_{post_id}"):
                                            update_feedback(post_id, 0)
                                            st.success("Feedback recorded. Thank you!")
                                    
                                    with col3_2:
                                        if st.button("👍 Real", key=f"real_{post_id}"):
                                            update_feedback(post_id, 1)
                                            st.success("Feedback recorded. Thank you!")
                                
                                # Additional post details
                                st.markdown(f"""
                                **Score**: {post['score']} | **Comments**: {post['num_comments']} | **Upvote Ratio**: {post['upvote_ratio']:.2f}
                                
                                **Clickbait**: {'Yes' if post['title_has_clickbait'] == 1 else 'No'} | **Sentiment**: {post['sentiment_compound']:.2f}
                                """)
                                
                                # Close the card div
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Add a small space between posts
                                st.markdown("<br>", unsafe_allow_html=True)
                        else:
                            st.error(f"Failed to fetch data from r/{subreddit_name}. Please try a different subreddit or time filter.")
            
            with tab2:
                # Single post analysis
                post_url = st.text_input("Enter Reddit post URL", 
                    placeholder="https://www.reddit.com/r/news/comments/...")
                
                if st.button("Analyze Post"):
                    if post_url:
                        with st.spinner("Analyzing post..."):
                            # Extract post ID from URL
                            post_id = extract_post_id_from_url(post_url)
                            
                            if post_id:
                                # Initialize Reddit API
                                reddit = initialize_reddit()
                                
                                # Fetch and analyze the post
                                post_data = analyze_single_post(reddit, post_id)
                                
                                if post_data is not None:
                                    # Make prediction
                                    result_data = predict_credibility(model, scaler, post_data)
                                    
                                    # Display result in a card
                                    post = result_data.iloc[0]
                                    
                                    # Determine card class based on credibility
                                    card_class = "poor-credibility" if post['probability_real'] < 0.3 else (
                                        "questionable-credibility" if post['probability_real'] < 0.7 else "good-credibility"
                                    )
                                    
                                    st.markdown(f"<div class='card {card_class}'>", unsafe_allow_html=True)
                                    
                                    # Post details
                                    st.markdown(f"**Title**: {post['title']}")
                                    st.markdown(f"**Domain**: {post['domain']} (Credibility Score: {post['domain_credibility_score']:.2f})")
                                    
                                    # Prediction results
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown(f"**Prediction**: {'❌ Potentially Fake' if post['is_fake_news'] else '✅ Likely Credible'}")
                                        st.markdown(f"**Confidence**: {max(post['probability_fake'], post['probability_real']):.2f}")  # Fixed format
                                    
                                    with col2:
                                        # Feedback buttons
                                        st.markdown("**Provide Feedback:**")
                                        col2_1, col2_2 = st.columns(2)
                                        
                                        with col2_1:
                                            if st.button("👎 Fake", key=f"fake_single_{post['id']}"):
                                                update_feedback(post['id'], 0)
                                                st.success("Feedback recorded. Thank you!")
                                        
                                        with col2_2:
                                            if st.button("👍 Real", key=f"real_single_{post['id']}"):
                                                update_feedback(post['id'], 1)
                                                st.success("Feedback recorded. Thank you!")
                                    
                                    # Additional metrics
                                    st.markdown(f"""
                                    **Score**: {post['score']} | **Comments**: {post['num_comments']} | **Upvote Ratio**: {post['upvote_ratio']:.2f}
                                    
                                    **Clickbait**: {'Yes' if post['title_has_clickbait'] == 1 else 'No'} | **Sentiment**: {post['sentiment_compound']:.2f}
                                    """)
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.error("Invalid Reddit post URL. Please make sure the URL is correct.")
                    else:
                        st.warning("Please enter a Reddit post URL.")

    # Model Performance page
    elif app_mode == "Model Performance":
        st.markdown("<h2 class='sub-header'>Model Performance Analysis</h2>", unsafe_allow_html=True)
        
        if not model_exists:
            st.warning("No trained model found. Please go to the 'Train Model' tab to train a model first.")
        else:
            # Load the model
            model = joblib.load("random_forest_model.joblib")
            scaler = joblib.load("scaler.joblib")
            
            # Load feedback data
            conn = setup_database()
            feedback_data = pd.read_sql_query(
                "SELECT * FROM training_data WHERE feedback IS NOT NULL", 
                conn
            )
            conn.close()
            
            # Display feedback stats
            if not feedback_data.empty:
                # Overall stats
                st.markdown("<h3 class='sub-header'>Feedback Statistics</h3>", unsafe_allow_html=True)
                
                total_feedback = len(feedback_data)
                fake_count = (feedback_data['feedback'] == 0).sum()
                real_count = (feedback_data['feedback'] == 1).sum()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Feedback", total_feedback)
                
                with col2:
                    st.metric("Marked as Fake", fake_count)
                
                with col3:
                    st.metric("Marked as Real", real_count)
                
                # Evaluate on feedback data
                st.markdown("<h3 class='sub-header'>Model Evaluation on Feedback Data</h3>", unsafe_allow_html=True)
                
                # Prepare features
                features = [
                    'is_self', 'score', 'upvote_ratio', 'num_comments', 'author_age_days',
                    'author_karma', 'sentiment_compound', 'word_count', 'avg_word_length',
                    'title_has_clickbait', 'credibility_score'
                ]
                
                X = feedback_data[features]
                y = feedback_data['feedback']
                
                # Handle missing values
                X.fillna(0, inplace=True)
                
                # Scale features
                X_scaled = scaler.transform(X)
                
                # Make predictions
                y_pred = model.predict(X_scaled)
                y_prob = model.predict_proba(X_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y, y_pred)
                report = classification_report(y, y_pred, output_dict=True)
                cm = confusion_matrix(y, y_pred)
                
                # Display metrics
                st.write(f"**Accuracy**: {accuracy:.4f}")
                
                # Confusion matrix
                st.markdown("<h4>Confusion Matrix</h4>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                ax.set_xticklabels(['Fake News', 'Credible'])
                ax.set_yticklabels(['Fake News', 'Credible'])
                st.pyplot(fig)
                
                # Classification report
                st.markdown("<h4>Classification Report</h4>", unsafe_allow_html=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
                # Feature importance
                st.markdown("<h3 class='sub-header'>Feature Importance</h3>", unsafe_allow_html=True)
                feature_importance = get_feature_importances(model, features)
                
                # Plot feature importances
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                ax.set_title('Feature Importance')
                st.pyplot(fig)
                
                # Learning curve
                st.markdown("<h3 class='sub-header'>Learning Progress</h3>", unsafe_allow_html=True)
                
                # Get timestamps of feedback entries for a timeline
                if 'created_utc' in feedback_data.columns:
                    # Sort by time
                    timeline_data = feedback_data.sort_values('created_utc')
                    
                    # Calculate cumulative accuracy over time
                    accuracies = []
                    timestamps = []
                    feedback_counts = []
                    
                    # Start with a minimum number of samples
                    min_samples = 10
                    
                    if len(timeline_data) >= min_samples:
                        for i in range(min_samples, len(timeline_data), max(1, len(timeline_data) // 20)):
                            subset = timeline_data.iloc[:i]
                            
                            X_subset = subset[features]
                            y_subset = subset['feedback']
                            
                            # Handle missing values
                            X_subset.fillna(0, inplace=True)
                            
                            # Scale features
                            X_subset_scaled = scaler.transform(X_subset)
                            
                            # Make predictions
                            y_subset_pred = model.predict(X_subset_scaled)
                            
                            # Calculate accuracy
                            subset_accuracy = accuracy_score(y_subset, y_subset_pred)
                            
                            accuracies.append(subset_accuracy)
                            timestamps.append(i)
                            feedback_counts.append(i)
                        
                        # Plot learning curve
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(feedback_counts, accuracies, 'o-', color='blue')
                        ax.set_xlabel('Number of Feedback Samples')
                        ax.set_ylabel('Accuracy')
                        ax.set_title('Learning Curve')
                        ax.grid(True)
                        st.pyplot(fig)
                    else:
                        st.info(f"Need at least {min_samples} feedback samples to display learning curve. Currently have {len(timeline_data)}.")
            else:
                st.info("No feedback data available yet. Provide feedback on posts in the 'Analyze Live Posts' tab.")
                
                # Option to generate synthetic data
                if st.button("Generate Synthetic Feedback Data"):
                    with st.spinner("Generating synthetic feedback data..."):
                        synthetic_data = generate_synthetic_training_data(min_samples=100)
                        st.success(f"Generated {len(synthetic_data)} synthetic data points.")
                        st.experimental_rerun()  # Rerun to update the interface

def extract_post_id_from_url(url):
    """Extract post ID from Reddit URL"""
    try:
        # Handle different Reddit URL formats
        if '/comments/' in url:
            return url.split('/comments/')[1].split('/')[0]
        return None
    except:
        return None

def analyze_single_post(reddit, post_id):
    """Fetch and analyze a single Reddit post"""
    try:
        # Fetch the post
        submission = reddit.submission(id=post_id)
        
        # Extract post data (similar to bulk analysis)
        if submission.author:
            author_age_days = (time.time() - submission.author.created_utc) / (60 * 60 * 24)
            author_karma = submission.author.comment_karma + submission.author.link_karma
        else:
            author_age_days = 30
            author_karma = 0
            
        domain = extract_domain(submission.url)
        domain_credibility = check_domain_credibility(domain)
        
        # Create post data dictionary
        post_data = {
            'id': [submission.id],
            'title': [submission.title],
            'domain': [domain],
            'is_self': [submission.is_self],
            'score': [submission.score],
            'upvote_ratio': [submission.upvote_ratio],
            'num_comments': [submission.num_comments],
            'author_age_days': [author_age_days],
            'author_karma': [author_karma],
            'credibility_score': [domain_credibility["credibility_score"]],
            'domain_credibility_score': [domain_credibility["credibility_score"]]
        }
        
        # Get sentiment
        combined_text = submission.title
        if submission.selftext:
            combined_text += " " + submission.selftext
        sentiment = sid.polarity_scores(combined_text)
        post_data['sentiment_compound'] = [sentiment["compound"]]
        
        # Get text complexity
        complexity = get_text_complexity(combined_text)
        post_data['word_count'] = [complexity["word_count"]]
        post_data['avg_word_length'] = [complexity["avg_word_length"]]
        
        # Check for clickbait
        post_data['title_has_clickbait'] = [check_clickbait(submission.title)]
        
        return pd.DataFrame(post_data)
    except Exception as e:
        st.error(f"Error fetching post: {str(e)}")
        return None

if __name__ == "__main__":
    main()
