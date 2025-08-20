#!/usr/bin/env bash
cd /home/tom/Repos/reddit-trends
source .venv/bin/activate
python3 tag_and_score.py >> logs/tag_and_score.log 2>&1
