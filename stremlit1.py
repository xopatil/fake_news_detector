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
import graphviz
from sklearn.tree import export_graphviz, plot_tree
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

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
    
    /* Decision Tree Analysis Styles */
    .tree-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    
    .feature-importance {
        background: linear-gradient(135deg, #f6f8fa 0%, #ffffff 100%);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .credible-post {
        border-left: 5px solid #28a745;
        background: linear-gradient(to right, #f0fff4, #ffffff);
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    
    .calculation-box {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }
    
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        text-transform: uppercase;
    }
    
    .decision-path {
        border-left: 3px dashed #6c757d;
        padding-left: 15px;
        margin: 10px 0;
    }
    
    .feature-highlight {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        background: rgba(0,123,255,0.1);
        color: #0056b3;
        font-family: monospace;
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
    
    # Add Mathematical Explanation in an expander
    with st.expander("View Domain Credibility Score Calculation", expanded=False):
        st.write("### 📐 Domain Credibility Score Calculation")
        
        # Base Score Formula
        st.subheader("1. Base Score Calculation")
        st.latex(r'''
        \text{Base Score} = \begin{cases}
        0.5 & \text{if domain is new} \\
        0.9 & \text{if domain in credible list} \\
        0.1 & \text{if domain in problematic list}
        \end{cases}
        ''')
        
        # API Score Formula
        st.subheader("2. API Score Calculation")
        st.latex(r'''
        \text{API Score} = \begin{cases}
        0.8 + \frac{\text{hash}(domain) \bmod 20}{100} & \text{if credible} \\
        0.1 + \frac{\text{hash}(domain) \bmod 30}{100} & \text{if problematic} \\
        0.3 + \frac{\text{hash}(domain)}{100} & \text{otherwise}
        \end{cases}
        ''')
        
        # Final Score Formula
        st.subheader("3. Final Score Determination")
        st.latex(r'''
        \text{Final Score} = \begin{cases}
        \text{API Score} & \text{if API available} \\
        \text{Base Score} & \text{if API fails}
        \end{cases}
        ''')
        
        st.write("""
        Where:
        - hash(domain) is MD5 hash modulo 100
        - Credible domains get scores between 0.8 and 1.0
        - Problematic domains get scores between 0.1 and 0.4
        - Unknown domains get scores between 0.3 and 0.7
        """)
        
        # Credibility Label Formula
        st.subheader("4. Credibility Label Assignment")
        st.latex(r'''
        \text{Credibility Label} = \begin{cases}
        \text{Credible} & \text{if score} > 0.7 \\
        \text{Questionable} & \text{if } 0.3 \leq \text{score} \leq 0.7 \\
        \text{Poor} & \text{if score} < 0.3
        \end{cases}
        ''')
        
        # Show current domain's calculation
        st.subheader("5. Current Domain Analysis")
        st.write(f"Domain being analyzed: **{domain}**")
        current_score = 0.5  # Neutral starting point
        st.write(f"Calculated credibility score: **{current_score:.2f}**")
        st.write(f"Credibility status: **{'Credible' if domain in credible_domains else 'Not Credible'}**")
        st.write(f"Problematic status: **{'Yes' if domain in problematic_domains else 'No'}**")
    
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
def fetch_reddit_data(subreddit_name, time_filter, limit=5000):
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
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each post individually
        for i, submission in enumerate(subreddit.new(limit=limit)):
            # Update progress
            progress = min(i / limit, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing post {i+1}/{limit}: {submission.title[:50]}...")
            
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
                        author_created_utc = time.time() - 30 * 86400
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
                    "author_age_days": author_age_days,
                    "author_karma": author_comment_karma + author_link_karma,
                    "credibility_score": domain_credibility.get("credibility_score", 0.5),
                    "domain_credibility_score": domain_credibility["credibility_score"],
                    "domain_is_credible": int(domain_credibility["is_credible"]),
                    "domain_is_problematic": int(domain_credibility["is_problematic"])
                }
                
                # Analyze sentiment from title and selftext
                combined_text = submission.title
                if submission.selftext:
                    combined_text += " " + submission.selftext
                    
                sentiment = sid.polarity_scores(combined_text)
                post_data.update({
                    "sentiment_neg": sentiment["neg"],
                    "sentiment_neu": sentiment["neu"],
                    "sentiment_pos": sentiment["pos"],
                    "sentiment_compound": sentiment["compound"]
                })
                
                # Get text complexity
                complexity = get_text_complexity(combined_text)
                post_data.update(complexity)
                
                # Check for clickbait
                post_data["title_has_clickbait"] = check_clickbait(submission.title)
                
                # Get comment sentiments
                submission.comments.replace_more(limit=0)
                comment_sentiments = []
                for comment in list(submission.comments)[:5]:
                    if comment.body:
                        comment_sentiment = sid.polarity_scores(comment.body)["compound"]
                        comment_sentiments.append(comment_sentiment)
                
                post_data["avg_comment_sentiment"] = np.mean(comment_sentiments) if comment_sentiments else 0
                post_data["comment_sentiment_variance"] = np.var(comment_sentiments) if comment_sentiments else 0
                
                # Store data in database
                data_for_db = {
                    "post_id": post_data["id"],
                    "title": post_data["title"],
                    "domain": post_data["domain"],
                    "is_self": int(post_data["is_self"]),
                    "score": post_data["score"],
                    "upvote_ratio": post_data["upvote_ratio"],
                    "num_comments": post_data["num_comments"],
                    "author_age_days": post_data["author_age_days"],
                    "author_karma": post_data["author_karma"],
                    "sentiment_compound": post_data["sentiment_compound"],
                    "word_count": post_data["word_count"],
                    "avg_word_length": post_data["avg_word_length"],
                    "title_has_clickbait": post_data["title_has_clickbait"],
                    "credibility_score": domain_credibility.get("credibility_score", 0.5),
                    "is_credible": domain_credibility.get("is_credible", 0),
                    "feedback": None
                }
                
                # Insert into database
                columns = ", ".join(data_for_db.keys())
                placeholders = ", ".join(["?"] * len(data_for_db))
                query = f"INSERT OR REPLACE INTO training_data ({columns}) VALUES ({placeholders})"
                c.execute(query, list(data_for_db.values()))
                conn.commit()
                
                all_posts_data.append(post_data)
                
            except Exception as e:
                st.error(f"Error processing post {submission.id}: {str(e)}")
                continue
        
        # Clear progress indicators
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
    # Load all training data, not just those with feedback
    query = "SELECT * FROM training_data WHERE credibility_score IS NOT NULL"
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
        st.write("### 🔍 Interactive Decision Tree Analysis")
        
        with st.expander("View Visual Decision Process", expanded=True):
            features = [
                'is_self', 'score', 'upvote_ratio', 'num_comments', 'author_age_days',
                'author_karma', 'sentiment_compound', 'word_count', 'avg_word_length',
                'title_has_clickbait', 'credibility_score'
            ]
            
            X_new = new_data[features].copy()
            X_new.fillna(0, inplace=True)
            X_new_scaled = scaler.transform(X_new)

            # Make predictions first
            predictions = model.predict(X_new_scaled)
            probabilities = model.predict_proba(X_new_scaled)
            
            # Update result data with predictions
            result_data = new_data.copy()
            result_data['is_fake_news'] = predictions
            result_data['probability_fake'] = [prob[0] for prob in probabilities]
            result_data['probability_real'] = [prob[1] for prob in probabilities]

            # Rest of the visualization code
            # Feature Importance Sunburst Chart
            st.write("### 1. Feature Importance Distribution")
            
            feature_importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            categories = {
                'Content': ['sentiment_compound', 'word_count', 'avg_word_length', 'title_has_clickbait'],
                'Engagement': ['score', 'upvote_ratio', 'num_comments'],
                'Source': ['is_self', 'credibility_score'],
                'Author': ['author_age_days', 'author_karma']
            }
            
            sunburst_data = []
            for category, feats in categories.items():
                sunburst_data.append({
                    'labels': category,
                    'parents': '',
                    'values': sum(feature_importance_df[feature_importance_df['Feature'].isin(feats)]['Importance'])
                })
                for feat in feats:
                    if feat in feature_importance_df['Feature'].values:
                        importance = feature_importance_df[feature_importance_df['Feature'] == feat]['Importance'].iloc[0]
                        sunburst_data.append({
                            'labels': feat,
                            'parents': category,
                            'values': importance
                        })
            
            sunburst_df = pd.DataFrame(sunburst_data)
            
            fig_sunburst = go.Figure(go.Sunburst(
                labels=sunburst_df['labels'],
                parents=sunburst_df['parents'],
                values=sunburst_df['values'],
                branchvalues="total",
                maxdepth=2,
                hovertemplate='<b>%{label}</b><br>Importance: %{value:.3f}<extra></extra>',
                ))
            
            fig_sunburst.update_layout(
                title={
                    'text': "Feature Importance Hierarchy",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                width=800,
                height=800,
                showlegend=False
            )
            
            st.plotly_chart(fig_sunburst, use_container_width=True)
            
            st.write("#### Feature Importance Table")
            importance_table = feature_importance_df.copy()
            importance_table['Importance'] = importance_table['Importance'].apply(lambda x: f"{x:.3f}")
            st.table(importance_table)

            # Decision Tree Paths
            st.write("### 2. Decision Tree Paths")
            
            # Get fresh training data (fixed query)
            conn = setup_database()
            query = """
            SELECT is_self, score, upvote_ratio, num_comments, author_age_days,
                   author_karma, sentiment_compound, word_count, avg_word_length,
                   title_has_clickbait, credibility_score, is_credible
            FROM training_data 
            WHERE credibility_score IS NOT NULL"""  # Simplified query to get all valid data
            fresh_training_data = pd.read_sql_query(query, conn)
            conn.close()

            if len(fresh_training_data) > 0:
                # Convert domain_is_credible to is_credible if needed
                if 'domain_is_credible' in fresh_training_data.columns:
                    fresh_training_data['is_credible'] = fresh_training_data['is_credible'].fillna(fresh_training_data['domain_is_credible'])
                
                # Prepare features for visualization trees
                X_viz = fresh_training_data[features].fillna(0)
                y_viz = fresh_training_data['is_credible'].astype(int)
                
                # Scale the features
                X_viz_scaled = scaler.transform(X_viz)
                
                # Create visualization trees
                viz_trees = []
                for _ in range(3):
                    tree = RandomForestClassifier(
                        n_estimators=1,
                        max_depth=3,
                        random_state=np.random.randint(0, 1000)
                    ).fit(X_viz_scaled, y_viz).estimators_[0]
                    viz_trees.append(tree)
                
                tree_tabs = st.tabs([f"Tree {i+1}" for i in range(3)])
                
                for idx, (tree, tab) in enumerate(zip(viz_trees, tree_tabs)):
                    with tab:
                        if os.path.exists(f"tree_{idx}.png"):
                            os.remove(f"tree_{idx}.png")
                        
                        dot_data = export_graphviz(
                            tree,
                            feature_names=features,
                            class_names=['Fake', 'Real'],
                            filled=True,
                            rounded=True,
                            special_characters=True,
                            max_depth=3,
                            proportion=True  # Show proportions instead of counts
                        )
                        graph = graphviz.Source(dot_data)
                        
                        # Generate new visualization
                        graph.render(f"tree_{idx}", format="png", cleanup=True)
                        
                        if os.path.exists(f"tree_{idx}.png"):
                            st.image(f"tree_{idx}.png", caption=f"Decision Path - Tree {idx + 1} (Sample size: {len(X_viz)})")
                        else:
                            st.error(f"Failed to generate tree visualization {idx + 1}")
            else:
                st.warning("No training data available for tree visualization")

            # Feature Interaction Analysis
            st.write("### 3. Feature Interaction Analysis")
            
            features_to_plot = feature_importance_df['Feature'][:4].tolist()
            fig = px.scatter_matrix(
                X_new[features_to_plot],
                dimensions=features_to_plot,
                title="Feature Relationships"
            )
            fig.update_layout(height=800)
            st.plotly_chart(fig)

            # Prediction Confidence Gauge
            st.write("### 4. Prediction Confidence")
            
            max_prob = max(probabilities[0])
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max_prob * 100,
                title={'text': f"Confidence in {'Real' if predictions[0] else 'Fake'} News"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen" if predictions[0] else "darkred"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig)

            # Tree Voting Distribution
            st.write("### 5. Ensemble Voting Distribution")
            
            tree_votes = [tree.predict(X_new_scaled)[0] for tree in model.estimators_]
            fake_votes = sum(1 for vote in tree_votes if vote == 0)
            real_votes = sum(1 for vote in tree_votes if vote == 1)
            
            fig = go.Figure(data=[go.Pie(
                labels=['Fake News', 'Real News'],
                values=[fake_votes, real_votes],
                hole=.3,
                marker=dict(colors=['#FF4B4B', '#00CC66'])
            )])
            fig.update_layout(
                title="Tree Voting Distribution",
                annotations=[dict(text=f'Total Trees<br>{len(tree_votes)}', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            st.plotly_chart(fig)

        return result_data

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        default_data = new_data.copy()
        default_data['is_fake_news'] = [0] * len(new_data)
        default_data['probability_fake'] = [0.0] * len(new_data)
        default_data['probability_real'] = [1.0] * len(new_data)
        return default_data

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
                                     ["Home", "Fetch Reddit Data", "Train Model", "Analyze Live Posts"])
    
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
        limit = st.slider("Number of posts to fetch", min_value=10, max_value=3500, value=100)  # Modified max_value to 1500
        
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
        
        # Create main tabs
        main_tabs = st.tabs(["Multiple Posts", "Single Post", "Credible Posts"])
        
        # Check if model exists and load it
        model_path = "random_forest_model.joblib"
        scaler_path = "scaler.joblib"
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                model_loaded = True
            else:
                st.warning("No trained model found. Please go to the 'Train Model' tab to train a model first.")
                model_loaded = False
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            model_loaded = False
            
        if not model_loaded:
            # Option to generate and train on synthetic data
            if st.button("Quick Start: Generate Data & Train Model"):
                with st.spinner("Generating synthetic data and training model..."):
                    # ...existing synthetic data generation code...
                    st.experimental_rerun()
        else:
            # Add Mathematical Formulas Section at the top
            with st.expander("📐 View Credibility Calculation Formulas", expanded=False):
                st.write("### Mathematical Formulas Used in Credibility Analysis")
                
                # Domain Credibility Score
                st.subheader("1. Domain Credibility Score")
                st.latex(r'''
                \text{Domain Score} = \begin{cases}
                0.9 & \text{if domain in trusted list} \\
                0.1 & \text{if domain in problematic list} \\
                0.5 + \frac{\text{hash}(domain)}{100} & \text{otherwise}
                \end{cases}
                ''')
                
                # Sentiment Analysis
                st.subheader("2. Sentiment Score")
                st.latex(r'''
                \text{Sentiment} = \frac{\text{positive} - \text{negative}}{\sqrt{\text{positive}^2 + \text{negative}^2}}
                ''')
                
                # Text Complexity
                st.subheader("3. Text Complexity")
                st.latex(r'''
                \text{Avg Word Length} = \frac{\sum \text{character count}}{\text{word count}}
                ''')
                
                # Final Credibility
                st.subheader("4. Overall Credibility")
                st.latex(r'''
                P(\text{credible}|X) = \frac{1}{T} \sum_{t=1}^{T} \text{Tree}_t(X)
                ''')
                st.write("Where T is the number of trees and X is the feature vector")

            # Create tabs for different analysis types
            with main_tabs[0]:  # Multiple Posts tab
                subreddit_name = st.text_input("Enter subreddit name (without r/)", "news")
                time_filter = st.selectbox("Select time filter", ["hour", "day", "week"])
                limit = st.slider("Number of posts to analyze", min_value=5, max_value=20, value=50)  # Modified max_value to 1500
                
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
                            
                            # Visualization of results (fixed histogram)
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            try:
                                # Check if we have enough unique values for a histogram
                                unique_values = result_data['probability_real'].nunique()
                                if unique_values > 1:
                                    # Use distplot for more stable density estimation
                                    sns.kdeplot(
                                        data=result_data, 
                                        x='probability_real',
                                        hue='is_fake_news',
                                        fill=True,
                                        ax=ax
                                    )
                                else:
                                    # Create a bar chart for limited data
                                    values = result_data['probability_real'].value_counts()
                                    ax.bar(values.index, values.values, 
                                          color=['green' if not result_data['is_fake_news'].iloc[0] else 'red'])
                                    ax.set_ylim(0, values.max() * 1.2)
                            except Exception as e:
                                # Fallback to simple bar plot
                                ax.bar(['Real News', 'Fake News'],
                                      [len(result_data[~result_data['is_fake_news']]),
                                       len(result_data[result_data['is_fake_news']])],
                                      color=['green', 'red'])
                            
                            ax.set_xlabel('Probability of being credible')
                            ax.set_ylabel('Density' if unique_values > 1 else 'Count')
                            ax.set_title('Distribution of Credibility Scores')
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
                            
                            # Store analysis results in session state
                            st.session_state['analysis_results'] = result_data
                            
                            # Credible Posts Analysis
                            with main_tabs[2]:  # Credible Posts Analysis
                                if 'analysis_results' in st.session_state:
                                    result_data = st.session_state['analysis_results']
                                    credible_posts = result_data[result_data['is_fake_news'] == 0].copy()
                                    
                                    st.markdown("### 🌟 Highly Credible Posts Analysis")
                                    st.write(f"Found {len(credible_posts)} credible posts")
                                    
                                    if len(credible_posts) > 0:
                                        credible_posts = credible_posts.sort_values('probability_real', ascending=False)
                                        
                                        # Update formula to match actual calculation
                                        st.markdown("### 📊 Confidence Score Calculation")
                                        st.write("""
                                        The overall confidence score is calculated as the model's probability 
                                        of the post being credible, while the component scores provide additional insight:
                                        """)
                                        
                                        for _, post in credible_posts.iterrows():
                                            with st.expander(f"{post['title'][:100]}..."):
                                                # Update to show both model confidence and component analysis
                                                st.markdown("#### Model Confidence")
                                                st.metric("Model Confidence Score", f"{post['probability_real']:.2f}")
                                                
                                                st.markdown("#### Component Analysis")
                                                # Calculate component scores (these are supplementary metrics)
                                                domain_weight = 0.3
                                                domain_score = post['credibility_score'] * domain_weight
                                                
                                                engagement_weight = 0.2
                                                engagement_score = (
                                                    (post['upvote_ratio'] * 0.7 + 
                                                     min(post['num_comments'], 1000)/1000 * 0.3)
                                                ) * engagement_weight
                                                
                                                content_weight = 0.3
                                                content_score = (
                                                    ((1 - post['title_has_clickbait']) * 0.4 + 
                                                     (post['sentiment_compound'] + 1)/2 * 0.3 +
                                                     min(post['word_count'], 500)/500 * 0.3)
                                                ) * content_weight
                                                
                                                author_weight = 0.2
                                                author_score = (
                                                    (min(post['author_age_days'], 1000)/1000 * 0.5 + 
                                                     min(post['author_karma'], 10000)/10000 * 0.5)
                                                ) * author_weight
                                                
                                                # Display component scores
                                                cols = st.columns(4)
                                                with cols[0]:
                                                    st.metric("Domain", f"{domain_score:.2f}")
                                                with cols[1]:
                                                    st.metric("Engagement", f"{engagement_score:.2f}")
                                                with cols[2]:
                                                    st.metric("Content", f"{content_score:.2f}")
                                                with cols[3]:
                                                    st.metric("Author", f"{author_score:.2f}")
                                                
                                                # Show component total
                                                component_total = domain_score + engagement_score + content_score + author_score
                                                st.markdown("#### Component Score Total")
                                                st.metric("Total Component Score", f"{component_total:.2f}")
                                                
                                                # Explain the difference
                                                st.info("""
                                                Note: The Model Confidence Score is based on the Random Forest model's prediction probability, 
                                                while the Component Score provides a complementary analysis of the post's credibility factors.
                                                These scores may differ as they measure credibility using different approaches.
                                                """)
                                                
                                                # Display raw metrics
                                                st.markdown("#### Raw Metrics")
                                                metrics_cols = st.columns(3)
                                                with metrics_cols[0]:
                                                    st.write(f"Domain Credibility: {post['credibility_score']:.2f}")
                                                    st.write(f"Upvote Ratio: {post['upvote_ratio']:.2f}")
                                                with metrics_cols[1]:
                                                    st.write(f"Comments: {post['num_comments']}")
                                                    st.write(f"Sentiment: {post['sentiment_compound']:.2f}")
                                                with metrics_cols[2]:
                                                    st.write(f"Author Age: {post['author_age_days']:.0f} days")
                                                    st.write(f"Author Karma: {post['author_karma']:,}")
                                    else:
                                        st.warning("No credible posts found in the current analysis.")
                            # ...existing code...
            
            with main_tabs[1]:  # Single Post tab
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
