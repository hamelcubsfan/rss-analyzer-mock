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

# Simple title
title = "📰 RSS Feed Analyzer"
st.title(title)
st.write("Discover insights across multiple RSS feeds using AI-powered analysis")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Set up Gemini API key from secrets
if hasattr(st.secrets, "gemini_api_key"):
    genai.configure(api_key=st.secrets.gemini_api_key)
else:
    st.error("Gemini API key not found in secrets. Please add it to your secrets.toml file.")
    st.stop()

# Model info
st.sidebar.markdown(
    """
🤖 Using Google's Gemini 2.5 Pro model
- Provides fast, high-quality responses
- Optimized for analysis tasks
"""
)
selected_model = "gemini-2.5-pro-exp-03-25"

# Default prompt
DEFAULT_PROMPT = """Analyze {feed_count} RSS news feeds related to the autonomous vehicle (AV) and robotics industries to produce a structured report on talent movement, leadership changes, layoffs, and related industry announcements. The report should be tailored for a recruiter at Waymo, focusing on hiring implications in key tech roles for AV development.

### OVERVIEW
Summarize key findings upfront, including:
- **Top Talent Movement Stories:** Major leadership changes, departures, or hires.
- **Industry Layoffs & Workforce Shifts:** Notable layoffs, hiring freezes, or workforce restructuring.
- **Emerging Trends & Impacts:** Recurring themes or broader shifts in talent movement affecting AV and robotics.
- **Recruitment Implications for Waymo:** Key takeaways and hiring opportunities.
---
... (rest of prompt) ..."""

# Prompt editor in sidebar
st.sidebar.subheader("Analysis Prompt")
user_prompt = st.sidebar.text_area(
    "Customize the analysis prompt:",
    value=DEFAULT_PROMPT,
    height=300,
    help="Customize how the AI analyzes the RSS feeds. Use {feed_count} as a placeholder for the number of feeds."
)

# Reset prompt button
if st.sidebar.button("Reset Prompt to Default"):
    user_prompt = DEFAULT_PROMPT
    st.sidebar.success("Prompt reset to default!")

# Scheduling options in sidebar
st.sidebar.subheader("Scheduled Analysis")
enable_scheduling = st.sidebar.checkbox("Enable Scheduled Analysis", value=False)
if enable_scheduling:
    schedule_timezone = st.sidebar.selectbox(
        "Select Timezone",
        options=pytz.common_timezones,
        index=pytz.common_timezones.index('US/Pacific')
    )
    morning_time = st.sidebar.time_input("Morning Analysis Time", datetime.strptime("08:00", "%H:%M"))
    evening_time = st.sidebar.time_input("Evening Analysis Time", datetime.strptime("17:00", "%H:%M"))

# NEW: Flexible feed parser
@st.cache_data(ttl=CACHE_TTL)
def parse_feeds(urls: List[str]) -> List[Dict[str, Any]]:
    all_entries: list[dict[str, Any]] = []

    for url in urls:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "rss-analyzer"})
            resp.raise_for_status()
        except Exception as e:
            st.warning(f"⚠️ couldn’t fetch {url}: {e}")
            continue

        raw = resp.content

        # RSS/Atom?
        if b"<rss" in raw[:512] or b"<feed" in raw[:512]:
            feed = feedparser.parse(raw)
            source = feed.feed.get("title", url)
            for ent in feed.entries:
                all_entries.append({
                    "feed_source": source,
                    "title": _clean_text(ent.get("title", "")),
                    "description": _clean_text(ent.get("description", "")),
                    "published": ent.get("published", "unknown")
                })
        else:
            all_entries.extend(_parse_html_river(raw, url))

    return all_entries


def _parse_html_river(html_bytes: bytes, source_url: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html_bytes, "lxml")
    river = soup.find("river")
    if not river:
        return []

    items = []
    for bullet in river.stripped_strings:
        segs = bullet.split("•", 1)
        if len(segs) != 2:
            continue
        time_part, rest = segs[0].strip(), segs[1].strip()

        m = re.match(r"(.+?)\s*/\s*(.+?):\s*(.+)", rest)
        if not m:
            continue
        author, source, headline = m.groups()

        items.append({
            "feed_source": source.strip(),
            "title": headline.strip(),
            "description": f"{author.strip()} – {headline.strip()}",
            "published": _normalize_techmeme_time(time_part)
        })
    return items


def _normalize_techmeme_time(t: str) -> str:
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        dt = datetime.strptime(f"{today} {t}", "%Y-%m-%d %I:%M %p")
        return dt.isoformat()
    except Exception:
        return t

# Original helpers (keep unchanged)
def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    return text.encode('utf-8', 'ignore').decode('utf-8')

# extract_content(), APIClient, initialize_session_state(), generate_summary(),
# load_analysis_history(), run_scheduled_analysis(), save_analysis_result(),
# scheduler setup, main() etc. remain exactly as in your original code.
