#!/usr/bin/env bash
cd /home/tom/Repos/reddit-trends
source .venv/bin/activate
python3 update_metrics.py >> logs/update_metrics.log 2>&1
