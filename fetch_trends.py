import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.deprecation")

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
SQLITE_TIMEOUT_SEC = int(os.environ.get("SQLITE_TIMEOUT_SEC", "30"))
SQLITE_BUSY_TIMEOUT_MS = int(os.environ.get("SQLITE_BUSY_TIMEOUT_MS", "60000"))
RETRY_MAX_ATTEMPTS = int(os.environ.get("RETRY_MAX_ATTEMPTS", "8"))
RETRY_BASE_DELAY_SEC = float(os.environ.get("RETRY_BASE_DELAY_SEC", "0.25"))
RETRY_BACKOFF = float(os.environ.get("RETRY_BACKOFF", "1.8"))
RETRY_MAX_DELAY_SEC = float(os.environ.get("RETRY_MAX_DELAY_SEC", "3.0"))

WINDOW_SIZES = [86400, 259200, 604800, 1209600]
MIN_CLUSTER_SIZE = int(os.environ.get("MIN_CLUSTER_SIZE", "5"))
STITCH_SIM_THRESHOLD = float(os.environ.get("STITCH_SIM_THRESHOLD", "0.9"))
STITCH_JACCARD_THRESHOLD = float(os.environ.get("STITCH_JACCARD_THRESHOLD", "0.3"))
PERSIST_MIN_WINDOWS = int(os.environ.get("PERSIST_MIN_WINDOWS", "2"))
COVERAGE_MAX_RATIO = float(os.environ.get("COVERAGE_MAX_RATIO", "0.5"))
CONCENTRATION_TOPK = int(os.environ.get("CONCENTRATION_TOPK", "2"))
CONCENTRATION_MIN_SHARE = float(os.environ.get("CONCENTRATION_MIN_SHARE", "0.6"))
PEAK_TO_MEDIAN_RATIO = float(os.environ.get("PEAK_TO_MEDIAN_RATIO", "1.5"))
MIN_UNIQUE_TAGS = int(os.environ.get("MIN_UNIQUE_TAGS", "3"))
SPIKE_MIN_SIZE = int(os.environ.get("SPIKE_MIN_SIZE", "20"))
SPECIFICITY_MIN = float(os.environ.get("SPECIFICITY_MIN", "0.4"))
GENERIC_RATE_LIMIT = float(os.environ.get("GENERIC_RATE_LIMIT", "0.6"))
LABEL_MODEL = os.environ.get("LABEL_MODEL", "gpt-4o")

class TrendLabel(BaseModel):
    label: str = Field(..., min_length=3, max_length=60)
    description: str = Field(..., min_length=6, max_length=160)

def connect_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=SQLITE_TIMEOUT_SEC)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    return con

def is_locked_error(e: Exception) -> bool:
    s = str(e).lower()
    return isinstance(e, sqlite3.OperationalError) and ("database is locked" in s or "database table is locked" in s or "database schema is locked" in s)

def run_with_retry(fn):
    delay = RETRY_BASE_DELAY_SEC
    last = None
    for _ in range(RETRY_MAX_ATTEMPTS):
        try:
            return fn()
        except Exception as e:
            last = e
            if is_locked_error(e):
                time.sleep(delay)
                delay = min(RETRY_MAX_DELAY_SEC, delay * RETRY_BACKOFF)
                continue
            raise
    if last:
        raise last

def execute_retry(cur: sqlite3.Cursor, sql: str, params: Tuple = ()) -> None:
    def op():
        cur.execute(sql, params)
    run_with_retry(op)

def executemany_retry(cur: sqlite3.Cursor, sql: str, seq) -> None:
    def op():
        cur.executemany(sql, seq)
    run_with_retry(op)

def fetchone_retry(cur: sqlite3.Cursor) -> Tuple:
    def op():
        return cur.fetchone()
    return run_with_retry(op)

def fetchall_retry(cur: sqlite3.Cursor) -> List[Tuple]:
    def op():
        return cur.fetchall()
    return run_with_retry(op)

def commit_retry(con: sqlite3.Connection) -> None:
    def op():
        con.commit()
    run_with_retry(op)

def ensure_trend_tables(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    execute_retry(cur, """
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
    execute_retry(cur, """
    CREATE TABLE IF NOT EXISTS trend_tags (
      trend_id TEXT NOT NULL,
      tag TEXT NOT NULL,
      count INTEGER NOT NULL,
      PRIMARY KEY (trend_id, tag),
      FOREIGN KEY (trend_id) REFERENCES trends(trend_id) ON DELETE CASCADE
    )
    """)
    commit_retry(con)

def fetch_time_range(con: sqlite3.Connection, subreddit: str) -> Tuple[int, int]:
    cur = con.cursor()
    execute_retry(cur, "SELECT MIN(created_utc), MAX(created_utc) FROM posts WHERE subreddit = ?", (subreddit,))
    row = fetchone_retry(cur)
    return (row[0] or 0, row[1] or 0)

def fetch_subreddits(con: sqlite3.Connection) -> List[str]:
    cur = con.cursor()
    execute_retry(cur, "SELECT DISTINCT subreddit FROM posts")
    return [r[0] for r in fetchall_retry(cur)]

def fetch_window_embeddings(con: sqlite3.Connection, subreddit: str, start_ts: int, end_ts: int) -> Tuple[List[str], np.ndarray]:
    cur = con.cursor()
    execute_retry(cur, """
        SELECT t.tag, e.embedding_json
        FROM tags t
        JOIN posts p ON p.post_id = t.post_id
        JOIN tag_embeddings e ON e.tag = t.tag
        WHERE p.subreddit = ?
          AND p.created_utc >= ?
          AND p.created_utc < ?
    """, (subreddit, start_ts, end_ts))
    rows = fetchall_retry(cur)
    if not rows:
        return [], np.empty((0,))
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
        counts: Dict[str, int] = {}
        for tg in d["tags"]:
            counts[tg] = counts.get(tg, 0) + 1
        d["tag_counts"] = counts
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
            if sim >= STITCH_SIM_THRESHOLD and jac >= STITCH_JACCARD_THRESHOLD:
                chain.append(nxt)
                used.add(j)
                last = nxt
        chains.append(chain)
    return chains

def concentration_share(counts: List[int], k: int) -> float:
    if not counts:
        return 0.0
    s = sum(counts)
    if s == 0:
        return 0.0
    top = sorted(counts, reverse=True)[:k]
    return sum(top) / s

def median_value(values: List[int]) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    if n % 2 == 1:
        return float(arr[n // 2])
    return float(arr[n // 2 - 1] + arr[n // 2]) / 2.0

def chain_specificity(summary_tags: Dict[str, int], presence: Dict[str, int], total_windows: int) -> float:
    if not summary_tags:
        return 0.0
    num = 0.0
    den = 0.0
    for tg, ct in summary_tags.items():
        pres = presence.get(tg, 0)
        rate = pres / max(1, total_windows)
        spec = 1.0 - min(1.0, rate)
        num += spec * ct
        den += ct
    if den == 0:
        return 0.0
    return num / den

def qualifies(chain: List[Dict], total_windows: int, presence: Dict[str, int]) -> bool:
    if not chain:
        return False
    counts = [c["size"] for c in chain]
    if len(chain) >= PERSIST_MIN_WINDOWS:
        cov = len(chain) / max(1, total_windows)
        if cov > COVERAGE_MAX_RATIO:
            return False
        conc = concentration_share(counts, CONCENTRATION_TOPK)
        if conc < CONCENTRATION_MIN_SHARE:
            return False
        med = median_value(counts)
        peak = max(counts)
        if med == 0:
            growth_ok = peak >= SPIKE_MIN_SIZE
        else:
            growth_ok = (peak / med) >= PEAK_TO_MEDIAN_RATIO
        if not growth_ok:
            return False
    else:
        c = chain[0]
        if c["size"] < SPIKE_MIN_SIZE:
            return False
    summary = summarize_chain(chain)
    spec = chain_specificity(summary["tags"], presence, total_windows)
    if spec < SPECIFICITY_MIN:
        return False
    if summary["unique"] < MIN_UNIQUE_TAGS:
        return False
    return True

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

def baseline_terms_from_presence(presence: Dict[str, int], total_windows: int, limit: int = 30) -> List[str]:
    scored = []
    for tg, pres in presence.items():
        rate = pres / max(1, total_windows)
        if rate >= GENERIC_RATE_LIMIT:
            scored.append((rate, tg))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:limit]]

def label_trend_llm(client: OpenAI, subreddit: str, tags: Dict[str, int], baseline_terms: List[str]) -> TrendLabel:
    top = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:25]
    tag_list = [f"{t}:{c}" for t, c in top]
    baseline = ", ".join(baseline_terms)
    sys = (
        "You name trends for an analytics dashboard. Use a plain descriptive label. "
        "Constraints: use one to five words. Prefer proper nouns only when clearly present in the tags such as model names or product names. "
        "Do not include the subreddit name. Do not produce labels that describe what the subreddit usually talks about. "
        "Avoid marketing words such as pulse, hub, nexus, surge, revolution, momentum, guru, vortex, fusion, mastery, mania, craze. "
        "Avoid made up compound words. Avoid cute phrasing. Keep capitalization minimal except for proper nouns. "
        "Return compact JSON with fields label and description."
    )
    user = (
        f"Subreddit for context only: {subreddit}\n"
        f"Baseline topics to ignore when naming: {baseline}\n"
        f"Tags with counts for this candidate trend: {', '.join(tag_list)}\n"
        f"Name a clear descriptive label and a one line description of what connects these tags."
    )
    try:
        resp = client.responses.parse(
            model=LABEL_MODEL,
            input=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            text_format=TrendLabel,
            temperature=0
        )
        return TrendLabel.model_validate(resp.output_parsed)
    except Exception:
        label = choose_fallback_label(tags)
        return TrendLabel(label=label, description="Automatic fallback label from top tags")

def clean_token(s: str) -> str:
    s = s.replace("_", " ").strip()
    s = " ".join(s.split())
    return s

def choose_fallback_label(tags: Dict[str, int]) -> str:
    if not tags:
        return ""
    scored = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:3]
    parts = [clean_token(t) for t, _ in scored]
    seen = set()
    final = []
    for t in parts:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        final.append(t)
    if not final:
        final = [clean_token(next(iter(tags.keys())))]
    return " ".join(final)

def insert_trend(con: sqlite3.Connection, subreddit: str, window_size: int, summary: Dict, lab: TrendLabel) -> str:
    tid = str(uuid.uuid4())
    cur = con.cursor()
    execute_retry(cur, """
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
    executemany_retry(cur, "INSERT OR REPLACE INTO trend_tags(trend_id, tag, count) VALUES(?,?,?)", rows)
    commit_retry(con)
    return tid

def main():
    con = connect_db(SQLITE_DB)
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
            presence: Dict[str, int] = {}
            total_windows = 0
            for s in starts:
                e = s + w
                if s >= tmax:
                    break
                tags, X = fetch_window_embeddings(con, sub, s, e)
                if tags:
                    total_windows += 1
                    seen = set(tags)
                    for tg in seen:
                        presence[tg] = presence.get(tg, 0) + 1
                if X.size == 0:
                    continue
                clusters = cluster_window(tags, X)
                for _, d in clusters.items():
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
            if not window_clusters or total_windows == 0:
                continue
            chains = stitch_clusters(window_clusters)
            baseline = baseline_terms_from_presence(presence, total_windows)
            for chain in chains:
                if not qualifies(chain, total_windows, presence):
                    continue
                summary = summarize_chain(chain)
                lab = label_trend_llm(client, sub, summary["tags"], baseline)
                insert_trend(con, sub, w, summary, lab)
    con.close()

if __name__ == "__main__":
    main()
