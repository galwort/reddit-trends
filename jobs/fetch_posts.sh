#!/usr/bin/env bash
cd /home/tom/Repos/reddit-trends
source .venv/bin/activate
py fetch_posts.py >> logs/fetch_posts.log 2>&1
