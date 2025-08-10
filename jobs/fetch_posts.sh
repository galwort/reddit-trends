#!/usr/bin/env bash
cd /home/tom/Repos/reddit-trends
source .venv/bin/activate
python3 fetch_posts.py >> logs/fetch_posts.log 2>&1
