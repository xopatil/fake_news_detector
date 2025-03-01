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

# Page configuration
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

# Function to extract domain from URL
def extract_domain(url):
    try:
        domain = urlparse(url).netloc
        return domain
    except:
        return ""

# Define global variables for domain lists
# These need to be declared before being referenced as global in functions
credible_domains = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", 
    "washingtonpost.com", "nytimes.com", "wsj.com", "economist.com",
    "hindustantimes.com", "indianexpress.com", "thehindu.com", "ndtv.com",
    "youtube.com"  # Added YouTube as it can contain credible content
]

problematic_domains = [
    "naturalnews.com", "infowars.com", "breitbart.com", "dailybuzzlive.com",
    "worldnewsdailyreport.com", "empirenews.net", "nationalreport.net"
]

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
        client_id="",
        client_secret="",
        user_agent=""
    )

# Fetch data from Reddit
def fetch_reddit_data(subreddit_name, time_filter, limit=100):
    reddit = initialize_reddit()
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        
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
        
        for submission in subreddit.new(limit=limit):
            if submission.created_utc >= start_timestamp:
                posts.append(submission)
        
        all_posts_data = []
        for post in posts:
            try:
                # Get author data 
                if post.author:
                    author_name = post.author.name
                    try:
                        author_created_utc = post.author.created_utc
                        author_age_days = (time.time() - author_created_utc) / (60 * 60 * 24)
                        author_comment_karma = post.author.comment_karma
                        author_link_karma = post.author.link_karma
                        author_has_verified_email = post.author.has_verified_email
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
                
                post_data = {
                    "id": post.id,
                    "title": post.title,
                    "url": post.url,
                    "domain": extract_domain(post.url),
                    "is_self": post.is_self,
                    "selftext": post.selftext,
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "created_utc": post.created_utc,
                    "post_age_days": (time.time() - post.created_utc) / (60 * 60 * 24),
                    "author": author_name,
                    "author_created_utc": author_created_utc,
                    "author_age_days": author_age_days,
                    "author_comment_karma": author_comment_karma,
                    "author_link_karma": author_link_karma,
                    "author_has_verified_email": author_has_verified_email,
                }
                
                # Analyze sentiment from title and selftext
                combined_text = post.title
                if post.selftext:
                    combined_text += " " + post.selftext
                    
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
                
                # Domain credibility
                domain = extract_domain(post.url)
                post_data["domain_is_credible"] = 1 if domain in credible_domains else 0
                post_data["domain_is_problematic"] = 1 if domain in problematic_domains else 0
                
                # Get top-level comments for sentiment analysis
                post_data["comments"] = []
                post.comments.replace_more(limit=0)  # Only get readily available comments
                comment_sentiments = []
                for comment in list(post.comments)[:5]:  # Get top 5 comments
                    if comment.body:
                        comment_sentiment = sid.polarity_scores(comment.body)["compound"]
                        comment_sentiments.append(comment_sentiment)
                        post_data["comments"].append({
                            "author": comment.author.name if comment.author else "[deleted]",
                            "body": comment.body,
                            "sentiment": comment_sentiment
                        })
                
                if comment_sentiments:
                    post_data["avg_comment_sentiment"] = np.mean(comment_sentiments)
                    post_data["comment_sentiment_variance"] = np.var(comment_sentiments)
                else:
                    post_data["avg_comment_sentiment"] = 0
                    post_data["comment_sentiment_variance"] = 0
                
                all_posts_data.append(post_data)
            except Exception as e:
                st.error(f"Error processing post: {str(e)}")
                continue
                
        return pd.DataFrame(all_posts_data)
    except Exception as e:
        st.error(f"Error fetching data from r/{subreddit_name}: {str(e)}")
        return pd.DataFrame()

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

# Load or train model
@st.cache_resource
def load_or_train_model():
    try:
        rf = load('reddit_fake_news_model.joblib')
        tfidf = load('reddit_tfidf_vectorizer.joblib')
        with open('model_columns.pkl', 'rb') as f:
            X_columns = pickle.load(f)
        return rf, tfidf, X_columns
    except:
        st.warning("Pre-trained model not found. Training a new model...")
        # Use sample data to train a basic model for demonstration
        return train_sample_model()

# Train a basic model using sample data
def train_sample_model():
    # Generate synthetic data
    sample_data = []
    
    # Credible post examples
    for i in range(30):
        sample_data.append({
            "id": f"cred{i}",
            "title": f"Latest economic report shows growth in manufacturing sector",
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
            "author_created_utc": time.time() - np.random.randint(365, 1825) * 86400,
            "author_age_days": np.random.randint(365, 1825),
            "author_comment_karma": np.random.randint(1000, 50000),
            "author_link_karma": np.random.randint(500, 10000),
            "author_has_verified_email": True,
            "sentiment_neg": np.random.uniform(0, 0.2),
            "sentiment_neu": np.random.uniform(0.5, 0.8),
            "sentiment_pos": np.random.uniform(0.1, 0.4),
            "sentiment_compound": np.random.uniform(0.1, 0.8),
            "word_count": np.random.randint(50, 300),
            "avg_word_length": np.random.uniform(4.5, 6.5),
            "sentence_count": np.random.randint(5, 30),
            "domain_is_credible": 1,
            "domain_is_problematic": 0,
            "avg_comment_sentiment": np.random.uniform(0.1, 0.5),
            "comment_sentiment_variance": np.random.uniform(0.05, 0.2)
        })
    
    # Fake news post examples
    for i in range(30):
        sample_data.append({
            "id": f"fake{i}",
            "title": f"SHOCKING: You won't believe what this politician did next!",
            "url": f"https://questionablenews{i}.com/article{i}",
            "domain": f"questionablenews{i}.com",
            "is_self": (i % 3 == 0),
            "selftext": "This incredible story has been suppressed by mainstream media!" if (i % 3 == 0) else "",
            "score": np.random.randint(5, 200),
            "upvote_ratio": np.random.uniform(0.4, 0.7),
            "num_comments": np.random.randint(5, 50),
            "created_utc": time.time() - np.random.randint(1, 15) * 86400,
            "post_age_days": np.random.randint(1, 15),
            "author": f"user{i+100}",
            "author_created_utc": time.time() - np.random.randint(30, 365) * 86400,
            "author_age_days": np.random.randint(30, 365),
            "author_comment_karma": np.random.randint(10, 1000),
            "author_link_karma": np.random.randint(5, 500),
            "author_has_verified_email": (i % 2 == 0),
            "sentiment_neg": np.random.uniform(0.1, 0.5),
            "sentiment_neu": np.random.uniform(0.3, 0.6),
            "sentiment_pos": np.random.uniform(0.1, 0.3),
            "sentiment_compound": np.random.uniform(-0.5, 0.3),
            "word_count": np.random.randint(20, 150),
            "avg_word_length": np.random.uniform(3.5, 5.5),
            "sentence_count": np.random.randint(3, 15),
            "domain_is_credible": 0,
            "domain_is_problematic": (i % 4 == 0),
            "avg_comment_sentiment": np.random.uniform(-0.3, 0.2),
            "comment_sentiment_variance": np.random.uniform(0.1, 0.4)
        })
    
    df = pd.DataFrame(sample_data)
    df = engineer_features(df)
    
    # Create synthetic labels
    credibility_score = (
        (df['domain_is_credible'] * 2) +
        (df['author_age_days'] / 365) +
        (df['author_comment_karma'] > 1000).astype(int) +
        (df['upvote_ratio'] > 0.8).astype(int) +
        (df['sentiment_compound'] > 0).astype(int) * 0.5 -
        (df['domain_is_problematic'] * 2) -
        (df['title_has_clickbait']) -
        (df['title_is_all_caps']) -
        ((df['comments_per_score'] > 1) & (df['score'] > 10)).astype(int) * 0.5
    )
    
    min_score = credibility_score.min()
    max_score = credibility_score.max()
    
    if max_score == min_score:
        df['credible'] = 1
    else:
        normalized_score = (credibility_score - min_score) / (max_score - min_score)
        df['credible'] = (normalized_score > 0.6).astype(int)
    
    # Prepare data for modeling
    features = [
        'score', 'upvote_ratio', 'num_comments', 'author_age_days', 
        'author_comment_karma', 'author_link_karma', 'sentiment_neg', 
        'sentiment_neu', 'sentiment_pos', 'sentiment_compound', 'word_count', 
        'avg_word_length', 'sentence_count', 'domain_is_credible', 
        'domain_is_problematic', 'avg_comment_sentiment', 
        'comment_sentiment_variance', 'title_has_clickbait', 'title_has_question', 
        'title_is_all_caps', 'title_selftext_ratio', 'post_hour', 'post_day',
        'comments_per_score'
    ]
    
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].fillna(0)
    y = df['credible']
    
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

# Predict credibility for a post
def predict_credibility(post_data, rf, tfidf, X_columns):
    # Create a single-row DataFrame with needed features
    df_new = pd.DataFrame([post_data])
    df_new = engineer_features(df_new)
    
    # Extract features that match the model's expected input
    features = [col for col in X_columns if col in df_new.columns]
    X_new_base = df_new[features].fillna(0)
    
    # Add TF-IDF features
    title_features = tfidf.transform([post_data["title"]])
    X_new = pd.concat([X_new_base.reset_index(drop=True), 
                      pd.DataFrame(title_features.toarray())], axis=1)
    
    # Ensure all expected columns are present
    missing_cols = set(X_columns) - set(X_new.columns)
    for col in missing_cols:
        X_new[col] = 0
        
    # Ensure correct column order
    X_new = X_new[X_columns]
    
    # Convert column names to strings
    X_new.columns = X_new.columns.astype(str)
    
    # Make prediction
    prediction = rf.predict(X_new)[0]
    probability = rf.predict_proba(X_new)[0][1]
    
    return {
        "is_credible": bool(prediction),
        "credibility_score": float(probability),
        "confidence": "high" if abs(probability - 0.5) > 0.3 else "medium" if abs(probability - 0.5) > 0.15 else "low"
    }

# Predict credibility for all posts in the DataFrame
def predict_all_posts(df, rf, tfidf, X_columns):
    # First engineer features for all posts
    df_engineered = engineer_features(df.copy())
    
    results = []
    for _, row in df_engineered.iterrows():
        result = predict_credibility(row.to_dict(), rf, tfidf, X_columns)
        results.append(result)
    
    df_results = pd.DataFrame(results)
    return pd.concat([df_engineered.reset_index(drop=True), df_results], axis=1)

# Main app UI
def main():
    st.title("Reddit News Credibility Analyzer")
    
    # Sidebar for inputs
    st.sidebar.header("Search Settings")
    
    subreddit_name = st.sidebar.text_input("Subreddit", value="news")
    
    time_filter = st.sidebar.selectbox(
        "Time Filter",
        options=["hour", "day", "week", "month", "year", "all"],
        index=1
    )
    
    limit = st.sidebar.slider("Maximum Posts to Analyze", 10, 200, 50)
    
    # Add custom domain lists
    st.sidebar.header("Custom Domain Lists")
    credible_domains = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", 
    "washingtonpost.com", "nytimes.com", "wsj.com", "economist.com",
    "hindustantimes.com", "indianexpress.com", "thehindu.com", "ndtv.com",
    "youtube.com"  # Added YouTube as it can contain credible content
    ]
    problematic_domains = [
    "naturalnews.com", "infowars.com", "breitbart.com", "dailybuzzlive.com",
    "worldnewsdailyreport.com", "empirenews.net", "nationalreport.net"
    ]
    # Using global keyword correctly
    with st.sidebar.expander("Manage Credible Domains"):
        custom_credible = st.text_area("Add credible domains (one per line)", 
                                      value="\n".join(credible_domains))
        if st.button("Update Credible Domains"):
            
            credible_domains = [d.strip() for d in custom_credible.split("\n") if d.strip()]
            st.success("Credible domains updated!")
    
    with st.sidebar.expander("Manage Problematic Domains"):
        custom_problematic = st.text_area("Add problematic domains (one per line)", 
                                        value="\n".join(problematic_domains))
        if st.button("Update Problematic Domains"):
            
            problematic_domains = [d.strip() for d in custom_problematic.split("\n") if d.strip()]
            st.success("Problematic domains updated!")
    
    # Load the model
    with st.spinner("Loading model..."):
        rf, tfidf, X_columns = load_or_train_model()
    
    # Fetch data button
    if st.sidebar.button("Fetch and Analyze"):
        with st.spinner(f"Fetching data from r/{subreddit_name}..."):
            df = fetch_reddit_data(subreddit_name, time_filter, limit)
        
        if not df.empty:
            st.success(f"Successfully fetched {len(df)} posts from r/{subreddit_name}")
            
            # Analyze posts
            with st.spinner("Analyzing posts for credibility..."):
                df_with_predictions = predict_all_posts(df, rf, tfidf, X_columns)
            
            # Display results
            st.header("Analysis Results")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            total_posts = len(df_with_predictions)
            credible_posts = df_with_predictions['is_credible'].sum()
            non_credible_posts = total_posts - credible_posts
            
            col1.metric("Total Posts Analyzed", total_posts)
            col2.metric("Credible Posts", credible_posts)
            col3.metric("Potentially Misleading Posts", non_credible_posts)
            
            # Display the results
            st.header("Analyzed Posts")
            
            # Filters
            filter_options = st.multiselect(
                "Filter by prediction:",
                options=["All", "Credible", "Potentially Misleading"],
                default=["All"]
            )
            
            # Sort options
            sort_by = st.selectbox(
                "Sort by:",
                options=["Newest First", "Oldest First", "Highest Score", "Most Comments", 
                         "Highest Credibility", "Lowest Credibility"]
            )
            
            # Apply filters
            filtered_df = df_with_predictions.copy()
            if "All" not in filter_options:
                if "Credible" in filter_options:
                    filtered_df = filtered_df[filtered_df['is_credible'] == True]
                if "Potentially Misleading" in filter_options:
                    filtered_df = filtered_df[filtered_df['is_credible'] == False]
            
            # Apply sorting
            if sort_by == "Newest First":
                filtered_df = filtered_df.sort_values("created_utc", ascending=False)
            elif sort_by == "Oldest First":
                filtered_df = filtered_df.sort_values("created_utc", ascending=True)
            elif sort_by == "Highest Score":
                filtered_df = filtered_df.sort_values("score", ascending=False)
            elif sort_by == "Most Comments":
                filtered_df = filtered_df.sort_values("num_comments", ascending=False)
            elif sort_by == "Highest Credibility":
                filtered_df = filtered_df.sort_values("credibility_score", ascending=False)
            elif sort_by == "Lowest Credibility":
                filtered_df = filtered_df.sort_values("credibility_score", ascending=True)
            
            # Display posts
            for _, post in filtered_df.iterrows():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.subheader(post['title'])
                        st.markdown(f"**Source:** {post['domain']}")
                        
                        # If it's a self post, show the text
                        if post['is_self'] and post['selftext']:
                            with st.expander("Show Content"):
                                st.write(post['selftext'])
                        else:
                            st.markdown(f"[View Original Post]({post['url']})")
                        
                        # Post metadata
                        st.text(f"Posted by u/{post['author']} • {datetime.fromtimestamp(post['created_utc']).strftime('%Y-%m-%d %H:%M:%S')} • {post['score']} points • {post['num_comments']} comments")
                    
                    with col2:
                        credibility_color = "green" if post['is_credible'] else "red"
                        credibility_label = "Credible" if post['is_credible'] else "Potentially Misleading"
                        credibility_percent = int(post['credibility_score'] * 100)
                        
                        st.markdown(f"<h3 style='color:{credibility_color};'>{credibility_label}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<p>Confidence: <b>{post['confidence']}</b></p>", unsafe_allow_html=True)
                        st.progress(credibility_percent)
                        st.text(f"Credibility score: {credibility_percent}%")
                    
                    # Key factors that influenced the prediction
                    with st.expander("View Factors"):
                        factors_col1, factors_col2 = st.columns(2)
                        
                        with factors_col1:
                            st.subheader("Positive Factors")
                            positive_factors = []
                            if post['domain_is_credible'] == 1:
                                positive_factors.append("• Source domain is considered credible")
                            if post['author_age_days'] > 365:
                                positive_factors.append("• Author account is over 1 year old")
                            if post['author_comment_karma'] > 1000:
                                positive_factors.append("• Author has significant comment karma")
                            if post['upvote_ratio'] > 0.8:
                                positive_factors.append("• High upvote ratio")
                            if post['sentiment_compound'] > 0.2:
                                positive_factors.append("• Positive sentiment in content")
                            if post['title_has_clickbait'] == 0:
                                positive_factors.append("• No clickbait elements in title")
                            if post['title_is_all_caps'] == 0:
                                positive_factors.append("• Title is not all caps")
                            
                            if positive_factors:
                                for factor in positive_factors:
                                    st.markdown(factor)
                            else:
                                st.text("No significant positive factors")
                        
                        with factors_col2:
                            st.subheader("Negative Factors")
                            negative_factors = []
                            if post['domain_is_credible'] == 0:
                                negative_factors.append("• Source domain is not in credible list")
                            if post['domain_is_problematic'] == 1:
                                negative_factors.append("• Source domain is known to be problematic")
                            if post['author_age_days'] < 30:
                                negative_factors.append("• Author account is less than 30 days old")
                            if post['title_has_clickbait'] == 1:
                                negative_factors.append("• Contains clickbait elements in title")
                            if post['title_is_all_caps'] == 1:
                                negative_factors.append("• Title is all caps")
                            if post['sentiment_compound'] < -0.2:
                                negative_factors.append("• Negative sentiment in content")
                            
                            if negative_factors:
                                for factor in negative_factors:
                                    st.markdown(factor)
                            else:
                                st.text("No significant negative factors")
                    
                    # Comments section
                    if 'comments' in post and post['comments']:
                        with st.expander("View Top Comments"):
                            for comment in post['comments']:
                                st.markdown(f"**u/{comment['author']}**")
                                st.text(comment['body'][:200] + "..." if len(comment['body']) > 200 else comment['body'])
                                st.text(f"Sentiment: {comment['sentiment']:.2f}")
                                st.divider()
                    
                    st.divider()
            
            # Download results option
            csv = filtered_df[['title', 'domain', 'score', 'num_comments', 'is_credible', 
                  'credibility_score', 'confidence']].to_csv(index=False)
            st.download_button(
                  label="Download Results as CSV",
                  data=csv,
                  file_name=f"reddit_credibility_analysis_{subreddit_name}.csv",
                  mime="text/csv"
              )
            
if __name__ == "__main__":
    main()