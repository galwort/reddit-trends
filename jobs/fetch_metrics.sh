#!/usr/bin/env bash
cd /home/tom/Repos/reddit-trends
source .venv/bin/activate
python3 fetch_metrics.py >> logs/fetch_metrics.log 2>&1
