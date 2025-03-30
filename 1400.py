import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import json
from datetime import datetime, timedelta
import praw
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.parse import urlparse
import os
import pickle
from joblib import load, dump
import requests
import hashlib
import sqlite3

st.set_page_config(
    page_title="Reddit News Credibility Analyzer",
    page_icon="📰",
    layout="wide"
)





# Download necessary NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        return True
    except:
        st.warning("NLTK download failed but continuing...")
        return False

download_nltk_data()

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Setup SQLite database for storing credibility data
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

# Function to extract domain from URL
def extract_domain(url):
    try:
        domain = urlparse(url).netloc
        # Remove 'www.' if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return ""

# Function to check domain credibility using external APIs and our database
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
        # MediaBiasFactCheck API (example - this is not a real API endpoint)
        # In a real implementation, you would use actual fact-checking APIs
        api_url = f"https://factcheck.example.api/{domain}"
        
        # This is a mock function to simulate an API call
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
    
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    
    word_count = len(words)
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    sentence_count = len(sentences)
    
    return {
        "word_count": word_count,
        "avg_word_length": avg_word_length,
        "sentence_count": sentence_count
    }

# Initialize Reddit API connection
@st.cache_resource
def initialize_reddit():
    return praw.Reddit(
        client_id="F2AyVPKAxPApH3arBONu9w",
        client_secret="06NmSk2W3V_nd2kllhnzsp1Oq3IRMA",
        user_agent="script:fakenews:1.0 (by u/Amazing-Bite-957)"
    )
def check_clickbait(title):
    """
    Checks if a post title has clickbait characteristics.
    
    Args:
        title (str): The title to analyze
        
    Returns:
        int: 1 if clickbait is detected, 0 otherwise
    """
    # Convert to lowercase for easier matching
    title_lower = title.lower()
    
    # Clickbait patterns
    clickbait_phrases = [
        "you won't believe", 
        "will shock you",
        "mind blowing",
        "shocking",
        "jaw-dropping",
        "jaw dropping",
        "amazing",
        "incredible",
        "unbelievable",
        "secret",
        "secrets",
        "they don't want you to know",
        "this one trick",
        "one simple trick",
        "life hack",
        "simple way",
        "easy way",
        "doctors hate",
        "doctors hate him",
        "doctors hate her",
        "what happens next",
        "what happened next",
        "the reason will",
        "the result will",
        "number 7 will",
        "find out why",
        "this is why"
    ]
    
    # Check for direct phrase matches
    for phrase in clickbait_phrases:
        if phrase in title_lower:
            return 1
    
    # Check for excessive punctuation or all caps
    if title.count('!') > 1 or title.count('?') > 2:
        return 1
    
    if sum(1 for c in title if c.isupper()) / max(len(title), 1) > 0.5:  # More than 50% uppercase
        return 1
    
    # Check for clickbait number patterns
    number_patterns = [
        r'\d+ (things|ways|tips|tricks|hacks|reasons|facts)',
        r'top \d+',
        r'\d+ (simple|easy|quick) (ways|steps|tips)'
    ]
    
    import re
    for pattern in number_patterns:
        if re.search(pattern, title_lower):
            return 1
    
    # Check for emotional bait with ellipsis or incomplete statements
    if title.endswith('...') or title.endswith('…'):
        return 1
    
    # Check for classic clickbait structures
    if (title_lower.startswith('when ') or title_lower.startswith('how ')) and '...' in title:
        return 1
    
    if (title_lower.startswith('this ') or title_lower.startswith('these ')) and any(word in title_lower for word in ['will', 'could', 'can', 'might']):
        return 1
        
    # Not detected as clickbait
    return 0

# Fetch data from Reddit
def fetch_reddit_data(subreddit_name, time_filter, limit=6000):
    reddit = initialize_reddit()
    conn = setup_database()
    c = conn.cursor()
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        
        if time_filter == "hour":
            start_time = datetime.utcnow() - timedelta(hours=1)
        elif time_filter == "day":
            start_time = datetime.utcnow() - timedelta(days=1)
        elif time_filter == "week":
            start_time = datetime.utcnow() - timedelta(weeks=1)
        elif time_filter == "month":
            start_time = datetime.utcnow() - timedelta(days=30)
        elif time_filter == "year":
            start_time = datetime.utcnow() - timedelta(days=365)
        else:  # All time
            start_time = datetime.utcfromtimestamp(0)
        
        start_timestamp = start_time.timestamp()
        
        # Collection for returning the data
        all_posts_data = []
        
        # Process each post individually
        for submission in subreddit.new(limit=limit):
            if submission.created_utc < start_timestamp:
                continue
                
            try:
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
                    "domain_credibility_score": domain_credibility["credibility_score"],
                    "domain_is_credible": int(domain_credibility["is_credible"]),
                    "domain_is_problematic": int(domain_credibility["is_problematic"])
                }
                
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
                
                # Check for clickbait (assuming there's a function for this)
                # If not available, it will use the default 0 from the get() method
                post_data["title_has_clickbait"] = check_clickbait(submission.title) if 'check_clickbait' in globals() else 0
                
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
                st.error(f"Error processing post {submission.id}: {str(e)}")
                continue
                
        return pd.DataFrame(all_posts_data)
        
    except Exception as e:
        st.error(f"Error fetching data from r/{subreddit_name}: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

# Feature engineering function
def engineer_features(df):
    if df.empty:
        return df
    
    # Fix the invalid escape sequence in regex
    df['title_has_clickbait'] = df['title'].str.contains('|'.join([
        'you won\'t believe', 'shocking', 'mind blowing', 'amazing', 'unbelievable', 
        'shocking truth', 'secret', 'revealed', 'this is why', 'this will', 'won\'t believe'
    ]), case=False, regex=True).astype(int)
    
    # Use raw string for regex pattern
    df['title_has_question'] = df['title'].str.contains(r'\?').astype(int)
    df['title_is_all_caps'] = df['title'].str.isupper().astype(int)
    
    df['title_selftext_ratio'] = 0.0
    for idx, row in df[df['is_self'] == True].iterrows():
        if row['selftext'] and row['title']:
            title_words = set(nltk.word_tokenize(row['title'].lower()))
            text_words = set(nltk.word_tokenize(row['selftext'].lower()))
            if title_words and text_words:
                overlap = len(title_words.intersection(text_words))
                df.at[idx, 'title_selftext_ratio'] = overlap / len(title_words)
    
    df['post_hour'] = df['created_utc'].apply(lambda x: datetime.fromtimestamp(x).hour)
    df['post_day'] = df['created_utc'].apply(lambda x: datetime.fromtimestamp(x).weekday())
    
    df['comments_per_score'] = df['num_comments'] / (df['score'] + 1)
    
    return df

# Load or train model with real data
@st.cache_resource
def load_or_train_model():
    try:
        rf = load('reddit_fake_news_model.joblib')
        tfidf = load('reddit_tfidf_vectorizer.joblib')
        with open('model_columns.pkl', 'rb') as f:
            X_columns = pickle.load(f)
        return rf, tfidf, X_columns
    except:
        st.warning("Pre-trained model not found. Training a model using stored data...")
        # Use data from our database if available, otherwise train a basic model
        conn = setup_database()
        df = pd.read_sql_query("SELECT * FROM training_data", conn)
        conn.close()
        
        if len(df) >= 20:  # Only use database if we have enough samples
            return train_model_from_database()
        else:
            return train_sample_model()

# Train model using data from our database
def train_model_from_database():
    conn = setup_database()
    df = pd.read_sql_query("SELECT * FROM training_data", conn)
    conn.close()
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare data for modeling
    features = [
        'score', 'upvote_ratio', 'num_comments', 'author_age_days', 
        'author_karma', 'sentiment_compound', 'word_count', 
        'avg_word_length', 'domain_is_credible', 
        'domain_is_problematic', 'title_has_clickbait', 'title_has_question', 
        'title_is_all_caps', 'title_selftext_ratio', 'post_hour', 'post_day',
        'comments_per_score'
    ]
    
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].fillna(0)
    y = df['is_credible']
    
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    title_features = tfidf.fit_transform(df['title'].fillna(''))
    X_with_text = pd.concat([X.reset_index(drop=True), pd.DataFrame(title_features.toarray())], axis=1)
    
    # Make sure all column names are strings
    X_with_text.columns = X_with_text.columns.astype(str)
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_with_text, y)
    
    # Save model for future use
    dump(rf, 'reddit_fake_news_model.joblib')
    dump(tfidf, 'reddit_tfidf_vectorizer.joblib')
    with open('model_columns.pkl', 'wb') as f:
        pickle.dump(X_with_text.columns, f)
    
    return rf, tfidf, X_with_text.columns

# Train a basic model using sample data - only used when no real data is available
def train_sample_model():
    # Generate synthetic data that's more realistic
    sample_data = []
    
    # Credible post examples
    for i in range(30):
        sample_data.append({
            "id": f"cred{i}",
            "title": f"New research shows economic growth in manufacturing sector",
            "url": f"https://reuters.com/article{i}",
            "domain": "reuters.com",
            "is_self": False,
            "selftext": "",
            "score": np.random.randint(50, 500),
            "upvote_ratio": np.random.uniform(0.7, 0.95),
            "num_comments": np.random.randint(10, 200),
            "created_utc": time.time() - np.random.randint(1, 30) * 86400,
            "post_age_days": np.random.randint(1, 30),
            "author": f"user{i}",
            "author_age_days": np.random.randint(365, 1825),
            "author_karma": np.random.randint(1000, 50000),
            "sentiment_compound": np.random.uniform(0.1, 0.8),
            "word_count": np.random.randint(50, 300),
            "avg_word_length": np.random.uniform(4.5, 6.5),
            "sentence_count": np.random.randint(5, 30),
            "domain_is_credible": 1,
            "domain_is_problematic": 0,
            "title_has_clickbait": 0,
            "title_has_question": 0,
            "title_is_all_caps": 0,
            "domain_credibility_score": np.random.uniform(0.7, 1.0)
        })
    
    # Add some credible posts from other domains
    credible_domains = ["washingtonpost.com", "nytimes.com", "bbc.com", "apnews.com"]
    for i in range(15):
        domain = credible_domains[i % len(credible_domains)]
        sample_data.append({
            "id": f"cred_other{i}",
            "title": f"New policy decisions impact global markets",
            "url": f"https://{domain}/article{i}",
            "domain": domain,
            "is_self": False,
            "selftext": "",
            "score": np.random.randint(30, 400),
            "upvote_ratio": np.random.uniform(0.65, 0.9),
            "num_comments": np.random.randint(8, 150),
            "created_utc": time.time() - np.random.randint(1, 30) * 86400,
            "post_age_days": np.random.randint(1, 30),
            "author": f"user{i+100}",
            "author_age_days": np.random.randint(180, 1500),
            "author_karma": np.random.randint(800, 30000),
            "sentiment_compound": np.random.uniform(0.0, 0.7),
            "word_count": np.random.randint(40, 250),
            "avg_word_length": np.random.uniform(4.3, 6.2),
            "sentence_count": np.random.randint(4, 25),
            "domain_is_credible": 1,
            "domain_is_problematic": 0,
            "title_has_clickbait": 0,
            "title_has_question": np.random.randint(0, 2),
            "title_is_all_caps": 0,
            "domain_credibility_score": np.random.uniform(0.7, 0.95)
        })
    
    # Fake news post examples
    problematic_domains = ["questionablenews.com", "dailybuzzlive.com", "infowars.com"]
    for i in range(30):
        domain = problematic_domains[i % len(problematic_domains)]
        sample_data.append({
            "id": f"fake{i}",
            "title": f"SHOCKING: You won't believe what this politician did next!",
            "url": f"https://{domain}/article{i}",
            "domain": domain,
            "is_self": (i % 3 == 0),
            "selftext": "This incredible story has been suppressed by mainstream media!" if (i % 3 == 0) else "",
            "score": np.random.randint(5, 200),
            "upvote_ratio": np.random.uniform(0.4, 0.7),
            "num_comments": np.random.randint(5, 50),
            "created_utc": time.time() - np.random.randint(1, 15) * 86400,
            "post_age_days": np.random.randint(1, 15),
            "author": f"user{i+200}",
            "author_age_days": np.random.randint(30, 365),
            "author_karma": np.random.randint(10, 1000),
            "sentiment_compound": np.random.uniform(-0.5, 0.3),
            "word_count": np.random.randint(20, 150),
            "avg_word_length": np.random.uniform(3.5, 5.5),
            "sentence_count": np.random.randint(3, 15),
            "domain_is_credible": 0,
            "domain_is_problematic": 1,
            "title_has_clickbait": 1,
            "title_has_question": np.random.randint(0, 2),
            "title_is_all_caps": np.random.randint(0, 2),
            "domain_credibility_score": np.random.uniform(0.05, 0.3)
        })
    
    # Add some unknown domain examples with mixed credibility
    for i in range(25):
        is_credible = np.random.randint(0, 2)
        clickbait_level = np.random.randint(0, 2)
        karma_level = np.random.randint(100, 5000) if is_credible else np.random.randint(50, 1000)
        
        sample_data.append({
            "id": f"mixed{i}",
            "title": f"{'BREAKING: ' if clickbait_level else ''}New study finds unexpected correlation in data" + ("?" if np.random.randint(0, 2) else ""),
            "url": f"https://news{i}.example.com/article{i}",
            "domain": f"news{i}.example.com",
            "is_self": (i % 5 == 0),
            "selftext": "Here are some interesting details about this finding..." if (i % 5 == 0) else "",
            "score": np.random.randint(10, 300),
            "upvote_ratio": np.random.uniform(0.5, 0.85),
            "num_comments": np.random.randint(5, 100),
            "created_utc": time.time() - np.random.randint(1, 20) * 86400,
            "post_age_days": np.random.randint(1, 20),
            "author": f"user{i+300}",
            "author_age_days": np.random.randint(60, 1000),
            "author_karma": karma_level,
            "sentiment_compound": np.random.uniform(-0.3, 0.6),
            "word_count": np.random.randint(30, 200),
            "avg_word_length": np.random.uniform(4.0, 6.0),
            "sentence_count": np.random.randint(3, 20),
            "domain_is_credible": 0,
            "domain_is_problematic": 0,
            "title_has_clickbait": clickbait_level,
            "title_has_question": np.random.randint(0, 2),
            "title_is_all_caps": 0,
            "domain_credibility_score": np.random.uniform(0.3, 0.7)
        })
    
    df = pd.DataFrame(sample_data)
    df = engineer_features(df)
    
    # Create more realistic labels based on multiple factors
    df['credibility_score'] = (
        (df['domain_credibility_score'] * 3) +  # Domain credibility is important
        (df['domain_is_credible'] * 1.5) - 
        (df['domain_is_problematic'] * 2) -
        (df['title_has_clickbait'] * 1.2) -
        (df['title_is_all_caps'] * 0.8) +
        (np.tanh(df['author_age_days'] / 365) * 0.7) +  # Diminishing returns on account age
        (np.tanh(df['author_karma'] / 5000) * 0.6) +    # Diminishing returns on karma
        (df['upvote_ratio'] - 0.5) * 2 +                # Scale upvote ratio impact
        (np.tanh(df['sentiment_compound'] + 0.3) * 0.5) # Slight penalty for very negative sentiment
    ) / 8  # Normalize to approximate 0-1 range
    
    # Clip to 0-1 range
    df['credibility_score'] = df['credibility_score'].clip(0, 1)
    
    # Create binary labels
    df['is_credible'] = (df['credibility_score'] > 0.6).astype(int)
    
    # Prepare data for modeling
    features = [
        'score', 'upvote_ratio', 'num_comments', 'author_age_days', 
        'author_karma', 'sentiment_compound', 'word_count', 
        'avg_word_length', 'domain_is_credible', 
        'domain_is_problematic', 'title_has_clickbait', 'title_has_question', 
        'title_is_all_caps', 'title_selftext_ratio', 'post_hour', 'post_day',
        'comments_per_score', 'domain_credibility_score'
    ]
    
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].fillna(0)
    y = df['is_credible']
    
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    title_features = tfidf.fit_transform(df['title'].fillna(''))
    X_with_text = pd.concat([X.reset_index(drop=True), pd.DataFrame(title_features.toarray())], axis=1)
    
    # Make sure all column names are strings
    X_with_text.columns = X_with_text.columns.astype(str)
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_with_text, y)
    
    # Save model for future use
    dump(rf, 'reddit_fake_news_model.joblib')
    dump(tfidf, 'reddit_tfidf_vectorizer.joblib')
    with open('model_columns.pkl', 'wb') as f:
        pickle.dump(X_with_text.columns, f)
    
    return rf, tfidf, X_with_text.columns

# Function to make predictions on new data

# Function to make predictions on new data
def predict_credibility(df, model, tfidf, X_columns):
    if df.empty:
        return df
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare the features
    features = [
        'score', 'upvote_ratio', 'num_comments', 'author_age_days', 
        'author_karma', 'sentiment_compound', 'word_count', 
        'avg_word_length', 'domain_is_credible', 
        'domain_is_problematic', 'title_has_clickbait', 'title_has_question', 
        'title_is_all_caps', 'title_selftext_ratio', 'post_hour', 'post_day',
        'comments_per_score'
    ]
    
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].fillna(0)
    
    # Extract text features
    title_features = tfidf.transform(df['title'].fillna(''))
    
    # Combine all features
    X_text_df = pd.DataFrame(title_features.toarray())
    X_text_df.columns = [str(col) for col in X_text_df.columns]
    X_with_text = pd.concat([X.reset_index(drop=True), X_text_df], axis=1)
    
    # Ensure we have all the columns the model was trained on
    for col in X_columns:
        if col not in X_with_text.columns:
            X_with_text[col] = 0
    
    # Keep only the columns used during training
    X_with_text = X_with_text[X_columns]
    
    # Make predictions
    df['credibility_prediction'] = model.predict(X_with_text)
    df['credibility_score'] = model.predict_proba(X_with_text)[:, 1]
    
    return df

# Function to store post data for model training
def store_post_data(post_data, user_feedback=None):
    conn = setup_database()
    c = conn.cursor()
    
    # Extract essential fields for storage
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
        "credibility_score": post_data.get("credibility_score", 0.5),
        "is_credible": post_data.get("credibility_prediction", 0),
        "feedback": user_feedback if user_feedback is not None else None
    }
    
    # Insert or update data
    placeholders = ', '.join(['?'] * len(data))
    columns = ', '.join(data.keys())
    values = tuple(data.values())
    
    query = f'''
    INSERT OR REPLACE INTO training_data
    ({columns})
    VALUES ({placeholders})
    '''
    
    try:
        # Execute the query with values
        c.execute(query, list(data.values()))
        conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

# Sidebar for app options
st.sidebar.title("Reddit News Credibility Analyzer")
analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    ["Analyze Subreddit", "Analyze Single Post"]
)
@st.cache_resource
def initialize_model():
    return load_or_train_model()

rf_model, tfidf_vectorizer, model_columns = initialize_model()

# Initialize model


if analysis_mode == "Analyze Subreddit":
    st.title("🔍 Reddit News Credibility Analyzer")
    st.write("Analyze news posts from Reddit to assess their credibility using machine learning.")
    
    col1, col2 = st.columns(2)
    with col1:
        subreddit = st.text_input("Enter subreddit name (e.g., news, worldnews)", "news")
    with col2:
        time_filter = st.selectbox(
            "Time period",
            ["day", "week", "month", "year", "all"]
        )
    
    post_limit = st.slider("Number of posts to analyze", 10, 500, 100)
    
    if st.button("Analyze Subreddit"):
        with st.spinner("Fetching and analyzing posts..."):
            df = fetch_reddit_data(subreddit, time_filter, post_limit)
            
            if not df.empty:
                df = predict_credibility(df, rf_model, tfidf_vectorizer, model_columns)
                
                # Display summary stats
                st.header(f"Analysis of r/{subreddit} - {len(df)} posts")
                
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                with metrics_col1:
                    credible_pct = df['credibility_prediction'].mean() * 100
                    st.metric("Credible Posts", f"{credible_pct:.1f}%")
                
                with metrics_col2:
                    avg_credibility = df['credibility_score'].mean() * 100
                    st.metric("Avg Credibility Score", f"{avg_credibility:.1f}%")
                
                with metrics_col3:
                    trusted_domains = df[df['domain_is_credible'] == 1]['domain'].nunique()
                    st.metric("Trusted Domains", trusted_domains)
                
                with metrics_col4:
                    problematic_domains = df[df['domain_is_problematic'] == 1]['domain'].nunique()
                    st.metric("Problematic Domains", problematic_domains)
                
                # Graph of credibility distribution
                st.subheader("Credibility Score Distribution")
                hist_data = np.histogram(
                    df['credibility_score'], 
                    bins=10, 
                    range=(0, 1)
                )
                hist_df = pd.DataFrame({
                    'Score Range': [f"{hist_data[1][i]:.1f}-{hist_data[1][i+1]:.1f}" for i in range(len(hist_data[1])-1)],
                    'Count': hist_data[0]
                })
                st.bar_chart(hist_df.set_index('Score Range'))
                
                # Top domains by count
                st.subheader("Top Domains")
                domain_counts = df['domain'].value_counts().reset_index()
                domain_counts.columns = ['Domain', 'Count']
                domain_counts['Average Credibility'] = domain_counts['Domain'].apply(
                    lambda d: df[df['domain'] == d]['credibility_score'].mean()
                )
                domain_counts = domain_counts.head(10)
                st.dataframe(domain_counts)
                
                # Display the posts with credibility info
                st.subheader("Analyzed Posts")
                
                # Add color-coding for credibility
                def highlight_credibility(row):
                  try:
                      score = float(row['credibility_score'].strip('%')) / 100  # Convert '13.0%' -> 0.13
                  except ValueError:
                      return [''] * len(row)  # If conversion fails, return no styling
                  
                  if score >= 0.8:
                      return ['background-color: #f0f0f0'] * len(row)  # Light gray
                  elif score >= 0.6:
                      return ['background-color: #d9d9d9'] * len(row)  # Medium gray
                  elif score >= 0.4:
                      return ['background-color: #bfbfbf'] * len(row)  # Darker gray
                  else:
                      return ['background-color: #808080; color: white'] * len(row)  # Dark gray with white text
                # Dark gray with white text

                
                display_df = df[['title', 'domain', 'score', 'num_comments', 'credibility_score']].copy()
                display_df['credibility_score'] = (display_df['credibility_score'] * 100).round(1).astype(str) + '%'
                display_df = display_df.style.apply(highlight_credibility, axis=1)
                
                st.dataframe(display_df)
                
                # Detailed analysis option
                st.subheader("Detailed Post Analysis")
                if "selected_post_idx" not in st.session_state:
                  st.session_state.selected_post_idx = 0  # Default to first post

                selected_post_idx = st.selectbox(
                    "Select a post for detailed analysis",
                    range(len(df)),
                    index=st.session_state.selected_post_idx,  # Use stored value
                    format_func=lambda i: df.iloc[i]['title'][:80] + "..."
                )

                st.session_state.selected_post_idx = selected_post_idx

                post = df.iloc[selected_post_idx]
                
                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    st.markdown(f"### {post['title']}")
                    st.write(f"**Domain:** {post['domain']}")
                    st.write(f"**Posted by:** u/{post['author']} (Account age: {post['author_age_days']:.1f} days)")
                    st.write(f"**Score:** {post['score']} (Upvote ratio: {post['upvote_ratio']:.2f})")
                    st.write(f"**Comments:** {post['num_comments']}")
                    
                    if post['is_self'] and post['selftext']:
                        st.write("**Post content:**")
                        st.text_area("", post['selftext'], height=150, disabled=True)
                
                with detail_col2:
                    # Credibility gauge chart
                    st.markdown("### Credibility Assessment")
                    credibility_score = post['credibility_score'] * 100
                    
                    if credibility_score >= 80:
                        credibility_color = "green"
                        credibility_text = "Highly Credible"
                    elif credibility_score >= 60:
                        credibility_color = "blue"
                        credibility_text = "Probably Credible"
                    elif credibility_score >= 40:
                        credibility_color = "orange"
                        credibility_text = "Uncertain"
                    else:
                        credibility_color = "red"
                        credibility_text = "Likely Not Credible"
                    
                    st.markdown(
                        f"<div style='text-align:center;'>"
                        f"<h1 style='color:{credibility_color};font-size:48px;'>{credibility_score:.1f}%</h1>"
                        f"<p style='color:{credibility_color};font-size:20px;'>{credibility_text}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Factors influencing score
                    st.markdown("### Key Factors")
                    
                    factors = []
                    if post['domain_is_credible'] == 1:
                        factors.append("✅ Trusted domain")
                    if post['domain_is_problematic'] == 1:
                        factors.append("❌ Problematic domain")
                    if post.get('title_has_clickbait', 0) == 1:
                        factors.append("❌ Clickbait title")
                    if post.get('title_is_all_caps', 0) == 1:
                        factors.append("❌ All-caps title")
                    if post['sentiment_compound'] < -0.5:
                        factors.append("⚠️ Highly negative sentiment")
                    if post['author_age_days'] < 90:
                        factors.append("⚠️ New account")
                    if post['author_age_days'] > 1000:
                        factors.append("✅ Established account")
                    
                    for factor in factors:
                        st.write(factor)
                
                # User feedback for improving the model
                st.subheader("Help Improve Our Model")
                st.write("Is our assessment correct? Provide feedback to help train the model.")
                
                feedback = st.radio(
                    "Is this post credible?",
                    ["Yes", "No", "Unsure"],
                    horizontal=True
                )
                
                if st.button("Submit Feedback"):
                    
                    feedback_map = {"Yes": 1, "No": 0, "Unsure": None}
                    store_post_data(post.to_dict(), feedback_map[feedback])
                    st.info("Retraining model with new feedback...")
                    
                    rf_model, tfidf_vectorizer, model_columns = load_or_train_model(force_retrain=True)
                    st.success("Thank you for your feedback! It will help improve our model.")

elif analysis_mode == "Analyze Single Post":
    st.title("🔍 Analyze a Specific Reddit Post")
    
    post_url = st.text_input("Enter the full URL to a Reddit post")
    
    if st.button("Analyze Post") and post_url:
        # Extract post ID from URL
        post_id_match = re.search(r'comments/([a-z0-9]+)/', post_url)
        
        if post_id_match:
            post_id = post_id_match.group(1)
            
            with st.spinner("Analyzing post..."):
                reddit = initialize_reddit()
                try:
                    # Fetch the post
                    submission = reddit.submission(id=post_id)
                    
                    # Extract all the needed data
                    domain = extract_domain(submission.url)
                    domain_credibility = check_domain_credibility(domain)
                    
                    # Get author data
                    if submission.author:
                        author_name = submission.author.name
                        try:
                            author_created_utc = submission.author.created_utc
                            author_age_days = (time.time() - author_created_utc) / (60 * 60 * 24)
                            author_comment_karma = submission.author.comment_karma
                            author_link_karma = submission.author.link_karma
                            author_karma = author_comment_karma + author_link_karma
                        except:
                            author_created_utc = time.time() - 30 * 86400
                            author_age_days = 30
                            author_comment_karma = 100
                            author_link_karma = 50
                            author_karma = 150
                    else:
                        author_name = "[deleted]"
                        author_created_utc = time.time() - 30 * 86400
                        author_age_days = 30
                        author_comment_karma = 100
                        author_link_karma = 50
                        author_karma = 150
                    
                    # Sentiment analysis
                    combined_text = submission.title
                    if submission.selftext:
                        combined_text += " " + submission.selftext
                    
                    sentiment = sid.polarity_scores(combined_text)
                    complexity = get_text_complexity(combined_text)
                    
                    # Create a single-row DataFrame with the post data
                    post_data = [{
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
                        "author_created_utc": author_created_utc,
                        "author_age_days": author_age_days,
                        "author_comment_karma": author_comment_karma,
                        "author_link_karma": author_link_karma,
                        "author_karma": author_karma,
                        "sentiment_neg": sentiment["neg"],
                        "sentiment_neu": sentiment["neu"],
                        "sentiment_pos": sentiment["pos"],
                        "sentiment_compound": sentiment["compound"],
                        "word_count": complexity["word_count"],
                        "avg_word_length": complexity["avg_word_length"],
                        "sentence_count": complexity["sentence_count"],
                        "domain_credibility_score": domain_credibility["credibility_score"],
                        "domain_is_credible": int(domain_credibility["is_credible"]),
                        "domain_is_problematic": int(domain_credibility["is_problematic"])
                    }]
                    
                    post_df = pd.DataFrame(post_data)
                    
                    # Make prediction
                    post_df = predict_credibility(post_df, rf_model, tfidf_vectorizer, model_columns)
                    post = post_df.iloc[0]
                    
                    # Display results
                    st.header("Post Analysis")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"### {post['title']}")
                        st.write(f"**URL:** [{post['url']}]({post['url']})")
                        st.write(f"**Posted by:** u/{post['author']} (Account age: {post['author_age_days']:.1f} days)")
                        st.write(f"**Score:** {post['score']} (Upvote ratio: {post['upvote_ratio']:.2f})")
                        st.write(f"**Comments:** {post['num_comments']}")
                        
                        if post['is_self'] and post['selftext']:
                            st.write("**Post content:**")
                            st.text_area("", post['selftext'], height=150, disabled=True)
                    
                    with col2:
                        # Credibility gauge
                        credibility_score = post['credibility_score'] * 100
                        
                        if credibility_score >= 80:
                            credibility_color = "green"
                            credibility_text = "Highly Credible"
                        elif credibility_score >= 60:
                            credibility_color = "blue"
                            credibility_text = "Probably Credible"
                        elif credibility_score >= 40:
                            credibility_color = "orange"
                            credibility_text = "Uncertain"
                        else:
                            credibility_color = "red"
                            credibility_text = "Likely Not Credible"
                        
                        st.markdown(
                            f"<div style='text-align:center;background-color:#f8f9fa;padding:20px;border-radius:10px;'>"
                            f"<h2>Credibility Score</h2>"
                            f"<h1 style='color:{credibility_color};font-size:48px;'>{credibility_score:.1f}%</h1>"
                            f"<p style='color:{credibility_color};font-size:20px;'>{credibility_text}</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    
                    # Domain information
                    st.header("Domain Analysis")
                    domain_col1, domain_col2 = st.columns(2)
                    
                    with domain_col1:
                        st.write(f"**Domain:** {post['domain']}")
                        if post['domain_is_credible']:
                            st.success("This domain is recognized as a credible source")
                        elif post['domain_is_problematic']:
                            st.error("This domain is flagged as potentially problematic")
                        else:
                            st.info("This domain has no specific credibility flags")
                    
                    with domain_col2:
                        st.write(f"**Domain Credibility Score:** {post['domain_credibility_score'] * 100:.1f}%")
                        
                        # Mock data about domain for illustration
                        st.write("**Domain Categories:** " + 
                                 ("News, Politics, Journalism" if post['domain_credibility_score'] > 0.7 else 
                                  "News, Entertainment, Social Media"))
                    
                    # Content analysis
                    st.header("Content Analysis")
                    
                    content_col1, content_col2 = st.columns(2)
                    with content_col1:
                        st.subheader("Text Analysis")
                        st.write(f"**Word Count:** {post['word_count']}")
                        st.write(f"**Average Word Length:** {post['avg_word_length']:.2f} characters")
                        st.write(f"**Sentiment:** {post['sentiment_compound']:.2f} " + 
                                 ("(Positive)" if post['sentiment_compound'] > 0.05 else 
                                  "(Negative)" if post['sentiment_compound'] < -0.05 else 
                                  "(Neutral)"))
                        
                        # Check for clickbait
                        if 'title_has_clickbait' in post and post['title_has_clickbait'] == 1:
                            st.warning("⚠️ Title contains potential clickbait phrases")
                        
                        # Check for question headlines
                        if 'title_has_question' in post and post['title_has_question'] == 1:
                            st.info("ℹ️ Title is phrased as a question")
                            
                        # Check for all caps
                        if 'title_is_all_caps' in post and post['title_is_all_caps'] == 1:
                            st.warning("⚠️ Title is written in ALL CAPS")
                    
                    with content_col2:
                        st.subheader("Engagement Metrics")
                        metrics_data = {
                            'Metric': ['Upvote Ratio', 'Comments per Score', 'Post Age'],
                            'Value': [
                                f"{post['upvote_ratio']:.2f}", 
                                f"{post['num_comments'] / max(1, post['score']):.2f}",
                                f"{post['post_age_days']:.1f} days"
                            ]
                        }
                        st.table(pd.DataFrame(metrics_data))
                        
                        # Account assessment
                        account_age = post['author_age_days']
                        if account_age < 30:
                            st.warning("⚠️ Posted by a very new account (< 30 days)")
                        elif account_age < 180:
                            st.info("ℹ️ Posted by a relatively new account (< 6 months)")
                        else:
                            st.success("✅ Posted by an established account")
                    
                    # Feedback for model improvement
                    st.header("Feedback")
                    st.write("Do you agree with our assessment? Help us improve our model.")
                    
                    feedback = st.radio(
                        "Is this post credible?",
                        ["Yes", "No", "Unsure"],
                        horizontal=True
                    )
                    
                    if st.button("Submit Feedback"):
                        feedback_map = {"Yes": 1, "No": 0, "Unsure": None}
                        store_post_data(post.to_dict(), feedback_map[feedback])
                        st.success("Thank you for your feedback! It will help improve our model.")
                    
                except Exception as e:
                    st.error(f"Error analyzing post: {str(e)}")
        else:
            st.error("Invalid Reddit post URL. Please enter a valid URL.")

# Add information about the app
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app analyzes Reddit news posts to help determine their credibility. "
    "It uses machine learning to evaluate various factors including domain "
    "reputation, user history, content sentiment, and engagement patterns."
)
st.sidebar.markdown("### Factors Considered")
st.sidebar.markdown(
    "- Source domain reputation\n"
    "- Account age and karma\n"
    "- Post engagement metrics\n"
    "- Text analysis (sentiment, complexity)\n"
    "- Clickbait detection\n"
)
def extract_domain(url):
    """Extract the domain from a URL"""
    domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if domain_match:
        return domain_match.group(1)
    return url

def check_domain_credibility(domain):
    """Check if a domain is in our known credible or problematic lists"""
    # Example credibility database - in production this would be more comprehensive
    credible_domains = [
        'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'npr.org', 
        'nytimes.com', 'wsj.com', 'washingtonpost.com', 'ft.com',
        'economist.com', 'theguardian.com', 'cnn.com', 'nbcnews.com',
        'cbsnews.com', 'abcnews.go.com', 'pbs.org', 'bloomberg.com'
    ]
    
    problematic_domains = [
        'infowars.com', 'breitbart.com', 'dailycaller.com', 'naturalnews.com',
        'thegatewaypundit.com', 'zerohedge.com', 'rt.com', 'sputniknews.com'
    ]
    
    is_credible = domain in credible_domains
    is_problematic = domain in problematic_domains
    
    # Calculate a credibility score based on domain
    if is_credible:
        score = 0.9
    elif is_problematic:
        score = 0.1
    else:
        # Neutral score for unknown domains
        score = 0.5
        
    return {
        "is_credible": is_credible,
        "is_problematic": is_problematic,
        "credibility_score": score
    }

def get_text_complexity(text):
    """Analyze text complexity metrics"""
    if not text or text.strip() == '':
        return {"word_count": 0, "avg_word_length": 0, "sentence_count": 0}
        
    # Count words
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    
    # Average word length
    if word_count > 0:
        avg_word_length = sum(len(word) for word in words) / word_count
    else:
        avg_word_length = 0
    
    # Count sentences
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    return {
        "word_count": word_count,
        "avg_word_length": avg_word_length,
        "sentence_count": sentence_count
    }

def initialize_reddit():
    """Initialize Reddit API connection"""
    # Replace with your API credentials
    reddit = praw.Reddit(
        client_id='F2AyVPKAxPApH3arBONu9w',
        client_secret='06NmSk2W3V_nd2kllhnzsp1Oq3IRMA',
        user_agent='script:fakenews:1.0 (by u/Amazing-Bite-957)',
        username='YOUR_USERNAME',  # Optional
        password='YOUR_PASSWORD'   # Optional
    )
    return reddit

def load_or_train_model(force_retrain=False):
    """Load existing model or train a new one
    
    Args:
        force_retrain (bool): If True, retrain the model regardless of existing model
        
    Returns:
        tuple: (model, vectorizer, model_columns)
    """
    try:
        # If force_retrain is True, raise FileNotFoundError to trigger retraining
        if force_retrain:
            raise FileNotFoundError
            
        # Try to load existing model
        with open('credibility_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            
        rf_model = model_data['model']
        tfidf_vectorizer = model_data['vectorizer']
        model_columns = model_data['columns']
        
        return rf_model, tfidf_vectorizer, model_columns
        
    except FileNotFoundError:
        # Train a new model if none exists or retraining is forced
        conn = setup_database()
        query = "SELECT * FROM training_data WHERE feedback IS NOT NULL"
        training_data = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(training_data) < 5:  # Lowered threshold to ensure model trains with minimal data
            # Not enough data to train, return a basic model
            rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
            tfidf_vectorizer = TfidfVectorizer(max_features=100)
            
            # Define default columns for the model
            model_columns = [
                'score', 'upvote_ratio', 'num_comments', 'author_age_days',
                'author_karma', 'sentiment_compound', 'word_count', 
                'avg_word_length', 'domain_is_credible', 'domain_is_problematic'
            ]
            
            # Fit on some dummy data
            dummy_X = pd.DataFrame(np.zeros((10, len(model_columns))), columns=model_columns)
            dummy_y = np.zeros(10)
            rf_model.fit(dummy_X, dummy_y)
            
            # Fit vectorizer on dummy data
            tfidf_vectorizer.fit_transform(['dummy text'])
            
            # Debug info
            print(f"Created basic model with {len(training_data)} training samples")
            
            return rf_model, tfidf_vectorizer, model_columns
            
        # Train model on existing data
        print(f"Training model with {len(training_data)} samples")
        
        # Select features
        features = [
            'score', 'upvote_ratio', 'num_comments', 'author_age_days',
            'author_karma', 'sentiment_compound', 'word_count', 
            'avg_word_length', 'is_self', 'title_has_clickbait'
        ]
        
        # Add domain credibility if available
        if 'domain_is_credible' in training_data.columns:
            features.append('domain_is_credible')
        if 'domain_is_problematic' in training_data.columns:
            features.append('domain_is_problematic')
            
        # Ensure all features exist
        for feature in features:
            if feature not in training_data.columns:
                training_data[feature] = 0
                
        X = training_data[features]
        y = training_data['feedback']
        
        # Create text vectorizer if we have enough samples
        tfidf_vectorizer = TfidfVectorizer(max_features=200)
        if len(training_data) >= 10:
            tfidf_vectorizer.fit_transform(training_data['title'])
        else:
            tfidf_vectorizer.fit_transform(['dummy text'])
        
        # Train model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # Save model
        model_data = {
            'model': rf_model,
            'vectorizer': tfidf_vectorizer,
            'columns': features
        }
        
        with open('credibility_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model trained and saved successfully")
        return rf_model, tfidf_vectorizer, features

def fetch_reddit_data(subreddit_name, time_filter, limit):
    """Fetch posts from a subreddit and analyze them"""
    try:
        reddit = initialize_reddit()
        subreddit = reddit.subreddit(subreddit_name)
        
        # Get posts
        all_posts = []
        for submission in subreddit.top(time_filter=time_filter, limit=limit):
            # Extract domain
            domain = extract_domain(submission.url)
            domain_credibility = check_domain_credibility(domain)
            
            # Get author data
            if submission.author:
                author_name = submission.author.name
                try:
                    author_created_utc = submission.author.created_utc
                    author_age_days = (time.time() - author_created_utc) / (60 * 60 * 24)
                    author_comment_karma = submission.author.comment_karma
                    author_link_karma = submission.author.link_karma
                except:
                    author_created_utc = time.time() - 30 * 86400
                    author_age_days = 30
                    author_comment_karma = 100
                    author_link_karma = 50
            else:
                author_name = "[deleted]"
                author_created_utc = time.time() - 30 * 86400
                author_age_days = 30
                author_comment_karma = 100
                author_link_karma = 50
            
            # Text analysis
            post_text = submission.title
            if submission.selftext:
                post_text += " " + submission.selftext
                
            sentiment = sid.polarity_scores(post_text)
            complexity = get_text_complexity(post_text)
            
            # Check for clickbait indicators in title
            title_lower = submission.title.lower()
            clickbait_phrases = ['you won\'t believe', 'shocking', 'jaw-dropping', 'mind-blowing', 
                                'unbelievable', 'incredible', 'what happens next', 'secret', 'reveals']
            title_has_clickbait = any(phrase in title_lower for phrase in clickbait_phrases)
            
            title_has_question = '?' in submission.title
            title_is_all_caps = submission.title.isupper()
            
            # Create post data dictionary
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
                "author_created_utc": author_created_utc,
                "author_age_days": author_age_days,
                "author_comment_karma": author_comment_karma,
                "author_link_karma": author_link_karma,
                "author_karma": author_comment_karma + author_link_karma,
                "sentiment_neg": sentiment["neg"],
                "sentiment_neu": sentiment["neu"],
                "sentiment_pos": sentiment["pos"],
                "sentiment_compound": sentiment["compound"],
                "word_count": complexity["word_count"],
                "avg_word_length": complexity["avg_word_length"],
                "sentence_count": complexity["sentence_count"],
                "domain_credibility_score": domain_credibility["credibility_score"],
                "domain_is_credible": int(domain_credibility["is_credible"]),
                "domain_is_problematic": int(domain_credibility["is_problematic"]),
                "title_has_clickbait": int(title_has_clickbait),
                "title_has_question": int(title_has_question),
                "title_is_all_caps": int(title_is_all_caps)
            }
            
            all_posts.append(post_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_posts)
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def predict_credibility(df, model, vectorizer, columns):
    """Predict credibility for posts with better fallbacks"""
    # Create features for prediction
    features = [
        'score', 'upvote_ratio', 'num_comments', 'author_age_days',
        'author_karma', 'sentiment_compound', 'word_count', 
        'avg_word_length'
    ]
    
    # Add domain features if available
    if 'domain_is_credible' in df.columns:
        features.append('domain_is_credible')
    if 'domain_is_problematic' in df.columns:
        features.append('domain_is_problematic')
    
    # Add text features if available
    if 'title_has_clickbait' in df.columns:
        features.append('title_has_clickbait')
    if 'title_has_question' in df.columns:
        features.append('title_has_question')
    if 'title_is_all_caps' in df.columns:
        features.append('title_is_all_caps')
    
    # Ensure all required features exist
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Make heuristic prediction (always run as fallback or blend)
    credibility_score = (
        df.get('domain_is_credible', 0) * 0.4 +
        (1 - df.get('domain_is_problematic', 0)) * 0.3 +
        (df['author_age_days'] > 180).astype(int) * 0.1 +
        (df['upvote_ratio'] > 0.8).astype(int) * 0.1 +
        (df['sentiment_compound'] > 0).astype(int) * 0.05 +
        (1 - df.get('title_has_clickbait', 0)) * 0.05
    )
    
    # Check if we have a valid model
    valid_model = True
    try:
        # Get available features for this model
        model_features = list(set(features).intersection(set(columns)))
        
        if len(model_features) < 3:  # Need at least a few features
            valid_model = False
            print("Not enough matching features for model prediction")
        else:
            # Attempt model prediction
            X = df[model_features]
            model_pred = model.predict_proba(X)[:, 1]
            # Blend model prediction with heuristic (more weight to model)
            df['credibility_score'] = 0.7 * model_pred + 0.3 * credibility_score
    except Exception as e:
        print(f"Error using model for prediction: {e}")
        valid_model = False
    
    # Fallback to heuristic if model failed
    if not valid_model:
        print("Using heuristic prediction only")
        df['credibility_score'] = credibility_score
    
    # Binary prediction (1 = credible, 0 = not credible)
    df['credibility_prediction'] = (df['credibility_score'] >= 0.6).astype(int)
    
    return df
