import os
import json
from datetime import datetime, timezone, timedelta
import yaml
from dotenv import load_dotenv
import praw
from azure.storage.blob import BlobServiceClient

def utc_parts(ts):
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y/%m/%d/%H")

def make_blob_path(pulled_at, subreddit, post_id):
    return f"{utc_parts(pulled_at)}/{subreddit}/{post_id}.json"

def _hours_from_window(cfg_val: str | None, explicit_hours: int | None) -> int:
    if explicit_hours and explicit_hours > 0:
        return int(explicit_hours)
    w = (cfg_val or "day").strip().lower()
    mapping = {
        "hour": 1,
        "day": 24,
        "week": 24 * 7,
        "month": 24 * 30,
        "year": 24 * 365,
        "all": 24 * 365 * 5,
    }
    return mapping.get(w, 24)


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
    hours_override = cfg.get("pull_window_hours")
    hours = _hours_from_window(window, int(hours_override) if hours_override else None)
    start_utc = int((datetime.now(tz=timezone.utc) - timedelta(hours=hours)).timestamp())
    pulled_at = int(datetime.now(tz=timezone.utc).timestamp())

    total = 0
    for sub in cfg["subreddits"]:
        sr = reddit.subreddit(sub)
        saved_sub = 0
        for p in sr.new(limit=None):
            created = int(getattr(p, "created_utc", 0) or 0)
            if created < start_utc:
                break
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
                    "created_utc": created,
                    "author": str(getattr(p, "author", "")) if getattr(p, "author", None) else "",
                    "link_flair_text": getattr(p, "link_flair_text", None),
                    "url": getattr(p, "url", None),
                },
            }
            blob_path = make_blob_path(pulled_at, f"r_{sub}", p.id)
            try:
                if not container.get_blob_client(blob_path).exists():
                    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    container.upload_blob(name=blob_path, data=data)
                    total += 1
                    saved_sub += 1
            except Exception as e:
                print(f"error for {p.id}: {e}")
        print(f"{sub}: saved {saved_sub} posts in last {hours}h")
    print(f"saved {total} posts total")

if __name__ == "__main__":
    main()
