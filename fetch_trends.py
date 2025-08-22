import os
import json
import time
import uuid
import math
import sqlite3
from typing import Dict, List, Tuple
import numpy as np
import hdbscan
from sklearn.preprocessing import normalize
from pydantic import BaseModel, Field
from openai import OpenAI

SQLITE_DB = os.environ.get("SQLITE_DB", "reddit_trends.db")
WINDOW_SIZES = [86400, 259200, 604800, 1209600]
MIN_CLUSTER_SIZE = int(os.environ.get("MIN_CLUSTER_SIZE", "5"))
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.9"))
JACCARD_THRESHOLD = float(os.environ.get("JACCARD_THRESHOLD", "0.3"))
PERSIST_MIN_WINDOWS = int(os.environ.get("PERSIST_MIN_WINDOWS", "2"))
GROWTH_MIN_RATIO = float(os.environ.get("GROWTH_MIN_RATIO", "1.3"))
SPIKE_MIN_SIZE = int(os.environ.get("SPIKE_MIN_SIZE", "15"))
SPIKE_MIN_PROB = float(os.environ.get("SPIKE_MIN_PROB", "0.8"))
LABEL_MODEL = os.environ.get("LABEL_MODEL", "gpt-4o")

class TrendLabel(BaseModel):
    label: str = Field(..., min_length=3, max_length=60)
    description: str = Field(..., min_length=6, max_length=160)

def ensure_trend_tables(con: sqlite3.Connection) -> None:
    cur = con.cursor()
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
      centroid_json TEXT NOT NULL,
      label TEXT,
      label_model TEXT,
      created_utc INTEGER NOT NULL
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
    con.commit()

def fetch_time_range(con: sqlite3.Connection, subreddit: str) -> Tuple[int, int]:
    cur = con.cursor()
    cur.execute("SELECT MIN(created_utc), MAX(created_utc) FROM posts WHERE subreddit = ?", (subreddit,))
    row = cur.fetchone()
    return (row[0] or 0, row[1] or 0)

def fetch_subreddits(con: sqlite3.Connection) -> List[str]:
    cur = con.cursor()
    cur.execute("SELECT DISTINCT subreddit FROM posts")
    return [r[0] for r in cur.fetchall()]

def fetch_window_embeddings(con: sqlite3.Connection, subreddit: str, start_ts: int, end_ts: int) -> Tuple[List[str], np.ndarray]:
    cur = con.cursor()
    cur.execute("""
        SELECT t.tag, e.embedding_json
        FROM tags t
        JOIN posts p ON p.post_id = t.post_id
        JOIN tag_embeddings e ON e.tag = t.tag
        WHERE p.subreddit = ?
          AND p.created_utc >= ?
          AND p.created_utc < ?
    """, (subreddit, start_ts, end_ts))
    rows = cur.fetchall()
    if not rows:
        return [], np.empty((0,)), 
    tags = []
    vecs = []
    for tag, emb_json in rows:
        v = np.array(json.loads(emb_json), dtype=np.float32)
        tags.append(tag)
        vecs.append(v)
    X = np.vstack(vecs) if vecs else np.empty((0,))
    return tags, X

def l2_normalize_matrix(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X
    return normalize(X, norm="l2")

def l2_normalize_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def cluster_window(tags: List[str], X: np.ndarray) -> Dict[int, Dict]:
    if X.size == 0:
        return {}
    Xn = l2_normalize_matrix(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean")
    labels = clusterer.fit_predict(Xn)
    probs = clusterer.probabilities_
    clusters: Dict[int, Dict] = {}
    for idx, lbl in enumerate(labels):
        if lbl == -1:
            continue
        d = clusters.setdefault(lbl, {"indices": [], "tags": [], "probs": []})
        d["indices"].append(idx)
        d["tags"].append(tags[idx])
        d["probs"].append(float(probs[idx]))
    for lbl, d in list(clusters.items()):
        vecs = Xn[d["indices"]]
        centroid = vecs.mean(axis=0)
        centroid = l2_normalize_vector(centroid)
        d["centroid"] = centroid
        d["size"] = len(d["indices"])
        d["mean_prob"] = float(np.mean(d["probs"])) if d["probs"] else 0.0
        d["tag_counts"] = {}
        for tg in d["tags"]:
            d["tag_counts"][tg] = d["tag_counts"].get(tg, 0) + 1
    return clusters

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

def stitch_clusters(window_clusters: List[Dict]) -> List[List[Dict]]:
    chains: List[List[Dict]] = []
    window_clusters.sort(key=lambda c: c["start"])
    used = set()
    for i, c in enumerate(window_clusters):
        if i in used:
            continue
        chain = [c]
        used.add(i)
        last = c
        for j in range(i + 1, len(window_clusters)):
            nxt = window_clusters[j]
            if nxt["subreddit"] != last["subreddit"]:
                continue
            if nxt["window_size"] != last["window_size"]:
                continue
            if nxt["start"] < last["end"]:
                continue
            sim = cosine_sim(last["centroid"], nxt["centroid"])
            jac = jaccard(set(last["tag_counts"].keys()), set(nxt["tag_counts"].keys()))
            if sim >= SIM_THRESHOLD and jac >= JACCARD_THRESHOLD:
                chain.append(nxt)
                used.add(j)
                last = nxt
        chains.append(chain)
    return chains

def qualifies(chain: List[Dict]) -> bool:
    if len(chain) >= PERSIST_MIN_WINDOWS:
        first = chain[0]["size"]
        last = chain[-1]["size"]
        if last >= max(1, int(math.ceil(first * GROWTH_MIN_RATIO))):
            return True
        if len(chain) >= 3:
            return True
    if len(chain) == 1:
        c = chain[0]
        if c["size"] >= SPIKE_MIN_SIZE and c["mean_prob"] >= SPIKE_MIN_PROB:
            return True
    return False

def summarize_chain(chain: List[Dict]) -> Dict:
    all_tags: Dict[str, int] = {}
    for c in chain:
        for tg, ct in c["tag_counts"].items():
            all_tags[tg] = all_tags.get(tg, 0) + ct
    centroid = np.mean(np.vstack([c["centroid"] for c in chain]), axis=0)
    centroid = l2_normalize_vector(centroid)
    mean_prob = float(np.mean([c["mean_prob"] for c in chain]))
    start_utc = min(c["start"] for c in chain)
    end_utc = max(c["end"] for c in chain)
    return {
        "tags": all_tags,
        "centroid": centroid,
        "mean_prob": mean_prob,
        "start": start_utc,
        "end": end_utc,
        "unique": len(all_tags),
        "count": sum(all_tags.values())
    }

def label_trend(client: OpenAI, subreddit: str, tags: Dict[str, int]) -> TrendLabel:
    top = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:25]
    tag_list = [f"{t}:{c}" for t, c in top]
    sys = "You are a trend namer. Produce a short label and a one line description."
    user = f"Subreddit: {subreddit}\nTags with counts: {', '.join(tag_list)}\nReturn JSON with fields label and description."
    resp = client.responses.parse(
        model=LABEL_MODEL,
        input=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        text_format=TrendLabel,
        temperature=0
    )
    return TrendLabel.model_validate(resp.output_parsed)

def insert_trend(con: sqlite3.Connection, subreddit: str, window_size: int, summary: Dict, lab: TrendLabel) -> str:
    tid = str(uuid.uuid4())
    cur = con.cursor()
    cur.execute("""
        INSERT INTO trends(trend_id, subreddit, window_size_seconds, start_utc, end_utc, tag_count, unique_tag_count, mean_probability, centroid_json, label, label_model, created_utc)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        tid,
        subreddit,
        int(window_size),
        int(summary["start"]),
        int(summary["end"]),
        int(summary["count"]),
        int(summary["unique"]),
        float(summary["mean_prob"]),
        json.dumps(summary["centroid"].tolist()),
        lab.label,
        LABEL_MODEL,
        int(time.time())
    ))
    rows = [(tid, tg, int(ct)) for tg, ct in summary["tags"].items()]
    cur.executemany("INSERT OR REPLACE INTO trend_tags(trend_id, tag, count) VALUES(?,?,?)", rows)
    con.commit()
    return tid

def main():
    con = sqlite3.connect(SQLITE_DB)
    con.execute("PRAGMA journal_mode=WAL")
    ensure_trend_tables(con)
    subs = fetch_subreddits(con)
    client = OpenAI()
    for sub in subs:
        tmin, tmax = fetch_time_range(con, sub)
        if not tmin or not tmax or tmax <= tmin:
            continue
        full_size = tmax - tmin + 1
        sizes = WINDOW_SIZES + [full_size]
        for w in sizes:
            step = max(1, w // 2)
            starts = list(range(tmin, tmax + 1, step))
            window_clusters = []
            for s in starts:
                e = s + w
                if s >= tmax:
                    break
                tags, X = fetch_window_embeddings(con, sub, s, e)
                if X.size == 0:
                    continue
                clusters = cluster_window(tags, X)
                for lbl, d in clusters.items():
                    window_clusters.append({
                        "subreddit": sub,
                        "window_size": w,
                        "start": s,
                        "end": e,
                        "centroid": d["centroid"],
                        "tag_counts": d["tag_counts"],
                        "mean_prob": d["mean_prob"],
                        "size": d["size"]
                    })
            if not window_clusters:
                continue
            chains = stitch_clusters(window_clusters)
            for chain in chains:
                if not qualifies(chain):
                    continue
                summary = summarize_chain(chain)
                lab = label_trend(client, sub, summary["tags"])
                insert_trend(con, sub, w, summary, lab)
    con.close()

if __name__ == "__main__":
    main()
