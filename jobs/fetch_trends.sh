#!/usr/bin/env bash
cd /home/tom/Repos/reddit-trends
source .venv/bin/activate
python3 fetch_trends.py >> logs/fetch_trends.log 2>&1
