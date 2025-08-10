import os
import time
import sqlite3
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import praw

def now_utc():
    return int(datetime.now(tz=timezone.utc).timestamp())

def main():
    load_dotenv()
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        username=os.environ["REDDIT_USERNAME"],
        password=os.environ["REDDIT_PASSWORD"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )
    db_path = os.environ.get("SQLITE_DB", "reddit_trends.db")
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    cur = con.cursor()
    lookback_hours = int(os.environ.get("REFRESH_LOOKBACK_HOURS", "48"))
    created_after = now_utc() - lookback_hours * 3600
    cur.execute(
        """
        SELECT post_id FROM posts_raw
        WHERE created_utc >= ?
        """,
        (created_after,),
    )
    ids = [row[0] for row in cur.fetchall()]
    ts = now_utc()
    total = 0
    for pid in ids:
        try:
            s = reddit.submission(id=pid)
            score = int(getattr(s, "score", 0) or 0)
            num_comments = int(getattr(s, "num_comments", 0) or 0)
            upvote_ratio = float(getattr(s, "upvote_ratio", 0.0) or 0.0)
            cur.execute(
                """
                INSERT INTO post_metrics(post_id, observed_utc, score, num_comments, upvote_ratio)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(post_id, observed_utc) DO UPDATE SET
                  score=excluded.score,
                  num_comments=excluded.num_comments,
                  upvote_ratio=excluded.upvote_ratio
                """,
                (pid, ts, score, num_comments, upvote_ratio),
            )
            total += 1
        except Exception as e:
            print(f"refresh error {pid} {e}")
            continue
    con.commit()
    con.close()
    print(f"refreshed {total} snapshots at {ts}")

if __name__ == "__main__":
    main()
