import praw
from django.shortcuts import render
from django.http import JsonResponse

from .ml_model import FakeNewsDetector
import json

# Initialize the FakeNewsDetector instance
# We'll use a global variable to avoid re-training the model on each request
detector = None

def get_detector():
    """Get or initialize the fake news detector model"""
    global detector
    if detector is None:
        detector = FakeNewsDetector()
    return detector

def verify_news(request):
    """API endpoint to verify news authenticity (exempt from CSRF)"""
    if request.method == 'POST':
        try:
            # Parse request data
            data = json.loads(request.body)
            title = data.get('title', '')
            url = data.get('url', '')
            
            # Get detector instance
            detector = get_detector()
            
            # Get prediction
            result = detector.predict(title, url)
            
            return JsonResponse({
                'success': True, 
                'result': result
            })
        except Exception as e:
            return JsonResponse({
                'success': False, 
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False, 
        'error': 'Only POST method is supported'
    })
def fetch_reddit_data(subreddit_name):
    """Fetch data from Reddit for a specific subreddit"""
    # Initialize a Read-Only Reddit Instance
    reddit = praw.Reddit(
        client_id="F2AyVPKAxPApH3arBONu9w",
        client_secret="06NmSk2W3V_nd2kllhnzsp1Oq3IRMA",
        user_agent="script:fakenews:1.0 (by u/Amazing-Bite-957)"
    )

    # Define subreddit
    subreddit = reddit.subreddit(subreddit_name)

    # Fetch posts
    post_data = []
    for post in subreddit.hot(limit=20):  # Limiting to 20 posts for performance
        post_data.append({
            "title": post.title,
            "url": post.url,
            "score": post.score,
            "comments": post.num_comments,
            "author": post.author.name if post.author else "Unknown",
            "created_utc": post.created_utc
        })
    
    return post_data

def home_view(request):
    """Render the home page with category buttons"""
    return render(request, 'reddit_fetcher/index.html')

def get_reddit_data(request, subreddit):
    """API endpoint to fetch Reddit data for a specific subreddit"""
    try:
        post_data = fetch_reddit_data(subreddit)
        return JsonResponse({'success': True, 'data': post_data})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})