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
from bs4 import BeautifulSoup

CACHE_TTL   = 300   # 5 minutes
MAX_RETRIES = 3
RETRY_DELAY = 1

# ── Streamlit page setup ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="RSS Feed Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📰 RSS Feed Analyzer")
st.write("Discover insights across multiple RSS feeds using AI-powered analysis")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Gemini key
if hasattr(st.secrets, "gemini_api_key"):
    genai.configure(api_key=st.secrets.gemini_api_key)
else:
    st.error("Gemini API key not found. Add it to .streamlit/secrets.toml")
    st.stop()

st.sidebar.markdown("""
🤖 Using Google's Gemini 2.5 Pro
""")
selected_model = "gemini-2.5-pro-exp-03-25"

# ── PROMPT etc. (unchanged) ───────────────────────────────────────────────────
DEFAULT_PROMPT = """Analyze {feed_count} RSS news feeds …"""  # (kept as-is)

st.sidebar.subheader("Analysis Prompt")
user_prompt = st.sidebar.text_area(
    "Customize the analysis prompt:",
    value=DEFAULT_PROMPT,
    height=300,
    help="Use {feed_count} as a placeholder for the number of feeds."
)
if st.sidebar.button("Reset Prompt to Default"):
    user_prompt = DEFAULT_PROMPT
    st.sidebar.success("Prompt reset to default!")

# ── Schedule UI (unchanged) ───────────────────────────────────────────────────
st.sidebar.subheader("Scheduled Analysis")
enable_scheduling = st.sidebar.checkbox("Enable Scheduled Analysis", value=False)
if enable_scheduling:
    schedule_timezone = st.sidebar.selectbox(
        "Select Timezone", options=pytz.common_timezones,
        index=pytz.common_timezones.index('US/Pacific')
    )
    morning_time = st.sidebar.time_input(
        "Morning Analysis Time",
        datetime.strptime("08:00", "%H:%M")
    )
    evening_time = st.sidebar.time_input(
        "Evening Analysis Time",
        datetime.strptime("17:00", "%H:%M")
    )

# ──────────────────────────────────────────────────────────────────────────────
#  ░ NEW ░  FLEXIBLE FEED PARSER  (RSS or Techmeme River HTML)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=CACHE_TTL)
def parse_feeds(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Returns a flat list of entries. Handles:
        • RSS / Atom  → via feedparser
        • Techmeme /river HTML  → custom _parse_html_river()
    """
    all_entries: list[dict[str, Any]] = []

    for url in urls:
        try:
            resp = requests.get(url, timeout=10,
                                headers={"User-Agent": "rss-analyzer"})
            resp.raise_for_status()
        except Exception as e:
            st.warning(f"⚠️ Couldn’t fetch {url}: {e}")
            continue

        raw = resp.content

        if b"<rss" in raw[:512] or b"<feed" in raw[:512]:
            # ---------- regular RSS ----------
            feed = feedparser.parse(raw)
            src  = feed.feed.get("title", url)
            for ent in feed.entries:
                all_entries.append({
                    "feed_source": src,
                    "title": _clean_text(ent.get("title", "")),
                    "description": _clean_text(ent.get("description", "")),
                    "published": ent.get("published", "unknown")
                })
        else:
            # ---------- HTML fallback ----------
            all_entries.extend(_parse_html_river(raw, url))

    return all_entries


def _parse_html_river(html_bytes: bytes, source_url: str) -> List[Dict[str, Any]]:
    """
    Extracts bullet lines from Techmeme’s /river page (inside <river>…</river>)
    and returns them in the same dict format used for RSS entries.
    """
    soup = BeautifulSoup(html_bytes, "lxml")
    river = soup.find("river")
    if not river:
        return []

    items = []
    for bullet in river.stripped_strings:
        # bullet looks like "7:55 PM  • Aisha Malik / TechCrunch:  Facebook will…"
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
    """Converts '7:55 PM' to ISO timestamp with today’s date."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        dt = datetime.strptime(f"{today} {t}", "%Y-%m-%d %I:%M %p")
        return dt.isoformat()
    except Exception:
        return t
# ──────────────────────────────────────────────────────────────────────────────

# ── Helper used elsewhere (unchanged) ─────────────────────────────────────────
def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    return text.encode('utf-8', 'ignore').decode('utf-8')

# ── extract_content(), APIClient, etc. (ALL UNCHANGED) ───────────────────────
# … (keep everything from your original file below this point exactly the same)
# ──────────────────────────────────────────────────────────────────────────────
