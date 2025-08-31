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
cur.execute("""
CREATE TABLE IF NOT EXISTS trends (
  trend_id TEXT PRIMARY KEY,
  subreddit TEXT NOT NULL,
  window_size_seconds INTEGER NOT NULL,
  start_utc INTEGER NOT NULL,
  end_utc INTEGER NOT NULL,
  tag_count INTEGER NOT NULL,
  unique_tag_count INTEGER NOT NULL,
  mean_probability REAL,
  label TEXT,
  label_model TEXT,
  created_utc INTEGER NOT NULL,
  coherence REAL,
  purity REAL,
  specificity REAL,
  watermark_id TEXT
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS trend_tags (
  trend_id TEXT NOT NULL,
  tag TEXT NOT NULL,
  count INTEGER NOT NULL,
  PRIMARY KEY (trend_id, tag),
  FOREIGN KEY (trend_id) REFERENCES trends(trend_id) ON DELETE CASCADE
)
""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_utc)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_post ON post_metrics(post_id, observed_utc DESC)")
con.commit()
con.close()
print(f"initialized {db_path}")
