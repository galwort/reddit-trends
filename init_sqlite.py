import os
import sqlite3

db_path = os.environ.get("SQLITE_DB", "reddit_trends.db")
con = sqlite3.connect(db_path)
cur = con.cursor()
cur.execute("PRAGMA journal_mode=WAL")
cur.execute("""
CREATE TABLE IF NOT EXISTS posts (
  post_id TEXT PRIMARY KEY,
  subreddit TEXT NOT NULL,
  title TEXT,
  selftext TEXT,
  author TEXT,
  permalink TEXT,
  link_flair_text TEXT,
  created_utc INTEGER NOT NULL,
  pulled_first_utc INTEGER NOT NULL,
  raw_pointer TEXT
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS post_metrics (
  post_id TEXT NOT NULL,
  observed_utc INTEGER NOT NULL,
  score INTEGER,
  num_comments INTEGER,
  upvote_ratio REAL,
  PRIMARY KEY (post_id, observed_utc),
  FOREIGN KEY (post_id) REFERENCES posts(post_id) ON DELETE CASCADE
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS tags (
  post_id TEXT NOT NULL,
  tag TEXT NOT NULL,
  sentiment REAL,
  confidence REAL,
  source TEXT,
  PRIMARY KEY (post_id, tag),
  FOREIGN KEY (post_id) REFERENCES posts(post_id) ON DELETE CASCADE
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS tag_embeddings (
  tag TEXT PRIMARY KEY,
  embedding_json TEXT NOT NULL,
  dim INTEGER NOT NULL,
  updated_utc INTEGER NOT NULL
)
""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_utc)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_post ON post_metrics(post_id, observed_utc DESC)")
con.commit()
con.close()
print(f"initialized {db_path}")
