import os
import json
import time
from datetime import datetime, timezone, timedelta
import sqlite3
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

def hour_prefixes(now_utc, lookback_hours):
    prefixes = []
    base = datetime.fromtimestamp(now_utc, tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    for i in range(lookback_hours):
        t = base - timedelta(hours=i)
        prefixes.append(t.strftime("%Y/%m/%d/%H"))
    return prefixes

def insert_post(con, doc, blob_path):
    p = doc["payload"]
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO posts_raw
        (post_id, subreddit, title, selftext, author, permalink, link_flair_text, created_utc, pulled_first_utc, raw_pointer)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            p.get("id"),
            doc.get("subreddit"),
            p.get("title") or "",
            p.get("selftext") or "",
            p.get("author") or "",
            doc.get("permalink") or "",
            p.get("link_flair_text"),
            int(p.get("created_utc") or 0),
            int(doc.get("pulled_at_epoch") or 0),
            blob_path,
        ),
    )

def upsert_metrics(con, doc, observed_utc):
    p = doc["payload"]
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO post_metrics(post_id, observed_utc, score, num_comments, upvote_ratio)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(post_id, observed_utc) DO UPDATE SET
          score=excluded.score,
          num_comments=excluded.num_comments,
          upvote_ratio=excluded.upvote_ratio
        """,
        (
            p.get("id"),
            int(observed_utc),
            int(p.get("score") or 0),
            int(p.get("num_comments") or 0),
            float(p.get("upvote_ratio") or 0),
        ),
    )

def main():
    load_dotenv()
    bsc = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
    container = bsc.get_container_client(os.environ["AZURE_CONTAINER"])
    db_path = os.environ.get("SQLITE_DB", "reddit_trends.db")
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    now_utc = int(datetime.now(tz=timezone.utc).timestamp())
    lookback_hours = int(os.environ.get("LOAD_LOOKBACK_HOURS", "6"))
    prefixes = hour_prefixes(now_utc, lookback_hours)
    total = 0
    for prefix in prefixes:
        for blob in container.list_blobs(name_starts_with=f"{prefix}/"):
            if not blob.name.endswith(".json"):
                continue
            data = container.download_blob(blob.name).readall()
            doc = json.loads(data.decode("utf-8"))
            insert_post(con, doc, blob.name)
            observed = doc.get("pulled_at_epoch", now_utc)
            upsert_metrics(con, doc, observed)
            total += 1
    con.commit()
    con.close()
    print(f"loaded or updated {total} records into sqlite")

if __name__ == "__main__":
    main()
