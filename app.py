import streamlit as st
import feedparser
import os
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import json
import pytz
import requests
from typing import List, Dict, Any
from functools import lru_cache
from time import sleep
import streamlit.components.v1 as components
import google.generativeai as genai
import base64
import tempfile
from fpdf import FPDF

# NEW IMPORTS
from bs4 import BeautifulSoup
import re

# Constants
CACHE_TTL = 300  # 5 minutes cache for feed data
MAX_RETRIES = 3
RETRY_DELAY = 1

# Set page configuration
st.set_page_config(
    page_title="RSS Feed Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📰 RSS Feed Analyzer")
st.write("Discover insights across multiple RSS feeds using AI-powered analysis")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Prompt editor in sidebar
DEFAULT_PROMPT = """Analyze {feed_count} RSS news feeds related to the autonomous vehicle (AV) and robotics industries..."""
st.sidebar.subheader("Analysis Prompt")
user_prompt = st.sidebar.text_area(
    "Customize the analysis prompt:",
    value=DEFAULT_PROMPT,
    height=250,
    help="Use {feed_count} as placeholder for the number of feeds."
)
if st.sidebar.button("Reset Prompt to Default"):
    user_prompt = DEFAULT_PROMPT
    st.sidebar.success("Prompt reset to default!")

# RSS feed URLs input
st.sidebar.subheader("RSS Feed URLs")
def load_default_feeds():
    return [
        "https://techcrunch.com/feed/",
        "https://www.techmeme.com/feed.xml"
    ]
rss_feeds = st.sidebar.text_area(
    "Enter one RSS or HTML URL per line:",
    value="\n".join(load_default_feeds()),
    height=150,
)
rss_feeds = [u.strip() for u in rss_feeds.splitlines() if u.strip()]

# Scheduling options in sidebar
st.sidebar.subheader("Scheduled Analysis")
enable_scheduling = st.sidebar.checkbox("Enable Scheduled Analysis")
if enable_scheduling:
    schedule_timezone = st.sidebar.selectbox(
        "Timezone:", pytz.common_timezones, index=pytz.common_timezones.index('US/Pacific')
    )
    morning_time = st.sidebar.time_input("Morning time", datetime.strptime("08:00", "%H:%M"))
    evening_time = st.sidebar.time_input("Evening time", datetime.strptime("17:00", "%H:%M"))

# Gemini setup
if hasattr(st.secrets, "gemini_api_key"):
    genai.configure(api_key=st.secrets.gemini_api_key)
else:
    st.error("Missing Gemini API key in secrets.toml")
    st.stop()

# Main UI: Run button and analysis
if st.button("Run Cross-Feed Analysis"):
    if not rss_feeds:
        st.error("Please enter at least one URL.")
    else:
        st.write(f"Analyzing {len(rss_feeds)} sources...")
        entries = parse_feeds(rss_feeds)
        st.write(f"Total entries parsed: {len(entries)}")
        content = extract_content(entries, max_entries=10)
        summary = generate_summary(content, len(rss_feeds), user_prompt)
        st.subheader("Analysis Result")
        st.write(summary)
        # Download options
        b64 = base64.b64encode(summary.encode()).decode()
        href = f"<a href='data:text/plain;base64,{b64}' download='analysis.txt'>Download TXT</a>"
        st.markdown(href, unsafe_allow_html=True)

# Scheduler logic (unchanged)
# ... rest of your existing code: parse_feeds, helpers, scheduling, main() ...
