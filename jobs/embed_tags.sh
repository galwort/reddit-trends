#!/usr/bin/env bash
cd /home/tom/Repos/reddit-trends
source .venv/bin/activate
python3 embed_tags.py >> logs/embed_tags.log 2>&1
