import praw
import json

# Initialize a Read-Only Reddit Instance
reddit = praw.Reddit(
    client_id="F2AyVPKAxPApH3arBONu9w",  # Replace with correct Client ID
    client_secret="06NmSk2W3V_nd2kllhnzsp1Oq3IRMA",  # Replace with correct Client Secret
    user_agent="script:fakenews:1.0 (by u/Amazing-Bite-957)"  # Ensure no spaces in username
)

# Define subreddit
subreddit_name = "news"
subreddit = reddit.subreddit(subreddit_name)

# Fetch top 10 posts
post_data = []
for post in subreddit.hot(limit=100):
    post_data.append({
        "title": post.title,
        "url": post.url,
        "score": post.score,
        "comments": post.num_comments,
        "author": post.author.name if post.author else "Unknown",
        "created_utc": post.created_utc
    })

# Save data to JSON file
output_filename = f"{subreddit_name}_reddit_posts.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(post_data, f, indent=4)

print(f"Scraping complete! Data saved to {output_filename}")