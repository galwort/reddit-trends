import os
import time
import sqlite3
from typing import List
from pydantic import BaseModel, Field, ValidationError, field_validator
from dotenv import load_dotenv
from openai import OpenAI


class TagResult(BaseModel):
    tag: str = Field(..., min_length=1, max_length=80)
    sentiment: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("tag")
    def clean_tag(cls, v):
        return v.strip()


class TagList(BaseModel):
    tags: List[TagResult] = Field(..., min_items=3, max_items=10)


def ensure_tables(con):
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tags (
          post_id TEXT NOT NULL,
          tag TEXT NOT NULL,
          sentiment REAL,
          confidence REAL,
          source TEXT,
          PRIMARY KEY (post_id, tag),
          FOREIGN KEY (post_id) REFERENCES posts(post_id) ON DELETE CASCADE
        )
    """
    )
    try:
        cur.execute("ALTER TABLE posts ADD COLUMN tagged_at_utc INTEGER")
    except sqlite3.OperationalError:
        pass
    con.commit()


def fetch_untagged(con, limit):
    cur = con.cursor()
    cur.execute(
        """
        SELECT p.post_id, p.title, p.selftext
        FROM posts p
        LEFT JOIN tags te ON te.post_id = p.post_id
        WHERE te.post_id IS NULL
        ORDER BY p.created_utc DESC
        LIMIT ?
    """,
        (limit,),
    )
    return cur.fetchall()


def upsert_tags(con, pid, tags):
    cur = con.cursor()
    now = int(time.time())
    for t in tags:
        cur.execute(
            "INSERT OR IGNORE INTO tags(post_id, tag, sentiment, confidence, source) VALUES(?,?,?,?,?)",
            (pid, t.tag, float(t.sentiment), float(t.confidence), "gpt-4o"),
        )
    cur.execute("UPDATE posts SET tagged_at_utc = ? WHERE post_id = ?", (now, pid))
    con.commit()


def main():
    load_dotenv()
    db_path = os.environ.get("SQLITE_DB", "reddit_trends.db")
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    ensure_tables(con)
    rows = fetch_untagged(con, limit=20)
    if not rows:
        print("no untagged posts")
        return
    client = OpenAI()
    for pid, title, body in rows:
        text = f"{title}\n\n{body or ''}"
        sys = (
            "You extract concise topical tags that a trend analyst would use."
            " Return between five and eight tags."
            " Each tag has sentiment from 0 to 1 where 0 is negative and 1 is positive."
            " Confidence is from 0 to 1."
            " Tags should be short canonical phrases."
        )
        user = f"Text:\n{text[:6000]}"
        try:
            resp = client.responses.parse(
                model="gpt-4o",
                input=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
                text_format=TagList,
                temperature=0,
            )
            parsed = resp.output_parsed
            taglist = TagList.model_validate(parsed)
            upsert_tags(con, pid, taglist.tags)
            print(f"tagged {pid} count {len(taglist.tags)}")
        except ValidationError as ve:
            print(f"validation failed {pid} {ve}")
        except Exception as e:
            print(f"error {pid} {e}")
    con.close()


if __name__ == "__main__":
    main()
