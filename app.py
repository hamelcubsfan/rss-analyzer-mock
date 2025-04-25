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

# NEW IMPORTS for HTML parsing
from bs4 import BeautifulSoup
import re

# Constants
CACHE_TTL = 300  # 5 minutes
MAX_RETRIES = 3
RETRY_DELAY = 1

# Streamlit setup
st.set_page_config(
    page_title="RSS Feed Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title & description
st.title("📰 RSS Feed Analyzer")
st.write("Discover insights across multiple RSS feeds and HTML sources using AI-powered analysis")

# Sidebar – Configuration
st.sidebar.header("⚙️ Configuration")

# Prompt editor
DEFAULT_PROMPT = """Analyze {feed_count} RSS/HTML sources and produce a structured report on talent movement, layoffs, and industry shifts. Customize for AV/robotics recruiting."""
st.sidebar.subheader("Analysis Prompt")
user_prompt = st.sidebar.text_area(
    "Customize the analysis prompt:",
    value=DEFAULT_PROMPT,
    height=200,
    help="Use {feed_count} as placeholder for number of sources."
)
if st.sidebar.button("Reset Prompt to Default"):
    user_prompt = DEFAULT_PROMPT
    st.sidebar.success("Prompt reset to default!")

# Feed URLs input
st.sidebar.subheader("Feed URLs")
def default_feeds():
    return [
        "https://techcrunch.com/feed/",
        "https://www.techmeme.com/river"
    ]
urls = st.sidebar.text_area(
    "Enter one RSS or HTML URL per line:",
    value="\n".join(default_feeds()),
    height=150
)
rss_feeds = [u.strip() for u in urls.splitlines() if u.strip()]

# Entries to analyze per source
max_entries = st.sidebar.number_input(
    "Entries per source:",
    min_value=1,
    max_value=50,
    value=10,
    help="How many items to extract per feed/source"
)

# Scheduled Analysis
st.sidebar.subheader("Scheduled Analysis")
enable_sched = st.sidebar.checkbox("Enable Scheduled Analysis")
if enable_sched:
    tz = st.sidebar.selectbox("Timezone", pytz.common_timezones, index=pytz.common_timezones.index('US/Pacific'))
    t1 = st.sidebar.time_input("Morning analysis at", datetime.strptime("08:00", "%H:%M"))
    t2 = st.sidebar.time_input("Evening analysis at", datetime.strptime("17:00", "%H:%M"))

# Gemini API key
if hasattr(st.secrets, "gemini_api_key"):
    genai.configure(api_key=st.secrets.gemini_api_key)
else:
    st.error("Missing Gemini API key in .streamlit/secrets.toml")
    st.stop()

# Flexible parser: RSS/Atom or HTML (Techmeme River)
@st.cache_data(ttl=CACHE_TTL)
def parse_feeds(urls: List[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "rss-analyzer"})
            resp.raise_for_status()
        except Exception as e:
            st.warning(f"Could not fetch {url}: {e}")
            continue
        raw = resp.content
        # detect RSS/Atom
        if b"<rss" in raw[:512] or b"<feed" in raw[:512]:
            feed = feedparser.parse(raw)
            src = feed.feed.get('title', url)
            for e0 in feed.entries:
                entries.append({
                    'feed_source': src,
                    'title': _clean_text(e0.get('title', '')),
                    'description': _clean_text(e0.get('description', '')),
                    'published': e0.get('published', '')
                })
        else:
            entries.extend(_parse_html_river(raw, url))
    return entries


def _parse_html_river(html: bytes, url: str) -> List[Dict[str, Any]]:
    # look for <river> tag or div#river
    soup = BeautifulSoup(html, 'lxml')
    river = soup.find('river') or soup.find(id='river')
    if not river:
        return []
    out: List[Dict[str, Any]] = []
    # split text by bullet char
    text = river.get_text(separator='\n')
    for line in text.split('\n'):
        if '•' not in line:
            continue
        parts = line.split('•', 1)
        t, rest = parts[0].strip(), parts[1].strip()
        m = re.match(r"(.+?)\s*/\s*(.+?):\s*(.+)", rest)
        if not m:
            continue
        auth, src, head = m.groups()
        out.append({
            'feed_source': src.strip(),
            'title': head.strip(),
            'description': f"{auth.strip()} – {head.strip()}",
            'published': _normalize_time(t)
        })
    return out


def _normalize_time(t: str) -> str:
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        dt = datetime.strptime(f"{today} {t}", "%Y-%m-%d %I:%M %p")
        return dt.isoformat()
    except:
        return t


def _clean_text(txt: str) -> str:
    if not isinstance(txt, str):
        return str(txt)
    reps = [("\u201c", '"'), ("\u201d", '"'), ("\u2018", "'"), ("\u2019", "'")]
    for a, b in reps:
        txt = txt.replace(a, b)
    return txt

# Summarization classes & functions
class APIClient:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
    def get_completion(self, prompt: str) -> str:
        resp = self.model.generate_content(prompt)
        return resp.text


def generate_summary(content: str, count: int, prompt: str) -> str:
    client = APIClient()
    full = prompt.format(feed_count=count) + "\n\n" + content
    return client.get_completion(full)

# Extract content
@st.cache_data(ttl=CACHE_TTL)
def extract_content(entries: List[Dict[str, Any]], max_entries: int) -> str:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        grouped.setdefault(e['feed_source'], []).append(e)
    parts: List[str] = []
    for src, items in grouped.items():
        parts.append(f"\nSource: {src}\n" + "-"*40)
        for it in items[:max_entries]:
            parts.append(f"Title: {it['title']}\nDate: {it['published']}\nDesc: {it['description']}\n")
    return "\n".join(parts)

# Analysis history helpers
```
