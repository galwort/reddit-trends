import os
import json
from datetime import datetime, timezone
import yaml
from dotenv import load_dotenv
import praw
from azure.storage.blob import BlobServiceClient

def utc_parts(ts):
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y/%m/%d/%H")

def make_blob_path(pulled_at, subreddit, post_id):
    return f"{utc_parts(pulled_at)}/{subreddit}/{post_id}.json"

def main():
    load_dotenv()
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        username=os.environ["REDDIT_USERNAME"],
        password=os.environ["REDDIT_PASSWORD"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )
    bsc = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
    container = bsc.get_container_client(os.environ["AZURE_CONTAINER"])
    window = str(cfg.get("pull_window", "day")).lower()
    limit = int(cfg.get("max_posts_per_sub", 50))
    pulled_at = int(datetime.now(tz=timezone.utc).timestamp())
    total = 0
    for sub in cfg["subreddits"]:
        sr = reddit.subreddit(sub)
        if window in {"hour", "day", "week", "month", "year", "all"}:
            posts = sr.top(time_filter=window, limit=limit)
        else:
            posts = sr.new(limit=limit)
        for p in posts:
            payload = {
                "source": "reddit",
                "pulled_at_epoch": pulled_at,
                "subreddit": f"r_{sub}",
                "post_id": p.id,
                "permalink": p.permalink,
                "payload": {
                    "id": p.id,
                    "title": p.title,
                    "selftext": getattr(p, "selftext", "") or "",
                    "score": int(getattr(p, "score", 0) or 0),
                    "upvote_ratio": float(getattr(p, "upvote_ratio", 0.0) or 0.0),
                    "num_comments": int(getattr(p, "num_comments", 0) or 0),
                    "created_utc": int(getattr(p, "created_utc", 0) or 0),
                    "author": str(getattr(p, "author", "")) if getattr(p, "author", None) else "",
                    "link_flair_text": getattr(p, "link_flair_text", None),
                    "url": getattr(p, "url", None)
                }
            }
            blob_path = make_blob_path(pulled_at, f"r_{sub}", p.id)
            try:
                if not container.get_blob_client(blob_path).exists():
                    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    container.upload_blob(name=blob_path, data=data)
                    total += 1
            except Exception as e:
                print(f"error for {p.id}: {e}")
    print(f"saved {total} posts")

if __name__ == "__main__":
    main()
