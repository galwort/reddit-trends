import os
import json
import time
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI

BATCH_SIZE = 500

def main():
    load_dotenv()
    db_path = "reddit_trends.db"
    model = "text-embedding-3-large"
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    cur = con.cursor()
    cur.execute("""
        SELECT DISTINCT t.tag
        FROM tags t
        LEFT JOIN tag_embeddings e ON e.tag = t.tag
        WHERE e.tag IS NULL
        LIMIT ?
    """, (BATCH_SIZE,))
    tags = [r[0] for r in cur.fetchall()]
    if not tags:
        print("no tags to embed")
        return
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=tags)
    now = int(time.time())
    rows = []
    for tag, item in zip(tags, resp.data):
        vec = item.embedding
        rows.append((tag, json.dumps(vec), len(vec), now))
    cur.executemany("INSERT OR REPLACE INTO tag_embeddings(tag, embedding_json, dim, updated_utc) VALUES(?,?,?,?)", rows)
    con.commit()
    con.close()
    print(f"embedded {len(rows)} tags")

if __name__ == "__main__":
    main()
