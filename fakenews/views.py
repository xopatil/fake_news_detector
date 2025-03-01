import praw
from django.shortcuts import render
from django.http import JsonResponse

from .ml_model import FakeNewsDetector
import json
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pickle
import os
from django.conf import settings
# Load the model at the top of the file
from .models import NewsDetector  # Adjust import based on your project structure

# Global variable to store model instance
detector = None


# Define the path
model_path = os.path.join(settings.BASE_DIR, 'fakenews', 'models', 'news_verification_model.pkl')


try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully:", model)
except Exception as e:
    print("❌ Error loading model:", e)
    model = None
# Load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)
@csrf_exempt
def verify_news(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            news_text = data.get("title", "")  # Extract title from request
            
            if not model:
                return JsonResponse({"error": "Model not loaded"}, status=500)

            if not news_text:
                return JsonResponse({"error": "No title provided"}, status=400)

            # Make a prediction
            prediction = model.predict([news_text])  # Ensure it's in a list format
            result = "Fake News" if prediction[0] == 1 else "Real News"

            return JsonResponse({"result": result})

        except Exception as e:
            print("❌ Error verifying news:", e)
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)

# Initialize the FakeNewsDetector instance
# We'll use a global variable to avoid re-training the model on each request
detector = None

def get_detector():
    """Get or initialize the fake news detector model"""
    global detector
    if detector is None:
        detector = FakeNewsDetector()
    return detector


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