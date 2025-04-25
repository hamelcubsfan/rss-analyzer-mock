import streamlit as st
import feedparser
import os
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import json
import pytz
import requests
from typing import List, Dict, Any
import google.generativeai as genai
import base64
from bs4 import BeautifulSoup
import re

# Constants
CACHE_TTL = 300  # seconds
MAX_RETRIES = 3

# Streamlit setup
st.set_page_config(
    page_title="RSS & HTML Feed Analyzer",
    page_icon="📰",
    layout="wide"
)

# Title & description
st.title("📰 RSS & HTML Feed Analyzer")
st.write("Aggregate and analyze news from RSS feeds or HTML pages with AI-powered summaries.")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Prompt editor
DEFAULT_PROMPT = """
Review the latest news from {feed_count} sources and produce a recruiter-focused report covering:

- Executive moves: promotions, departures, leadership changes.
- Workforce updates: layoffs, hiring freezes, major restructuring.
- Industry trends: funding rounds, partnerships, market shifts.
- Recruitment opportunities: in-demand roles & skills with sourcing recommendations.

### Structure
1) Overview: key highlights.
2) Detailed sections matching the bullets above.
Each entry: Title, Source, Time, Key details & recruiting impact.
"""
user_prompt = st.sidebar.text_area("Analysis prompt:", value=DEFAULT_PROMPT, height=200)
if st.sidebar.button("Reset prompt to default"):
    user_prompt = DEFAULT_PROMPT
    st.sidebar.success("Prompt reset")

# URL input
def default_urls():
    return [
        "https://techcrunch.com/feed/",
        "https://www.techmeme.com/river"
    ]
urls_text = st.sidebar.text_area(
    "Enter RSS or HTML URLs (one per line):",
    value="\n".join(default_urls()),
    height=150
)
urls = [u.strip() for u in urls_text.splitlines() if u.strip()]

# Entries per source
max_entries = st.sidebar.number_input("Entries per source:", min_value=1, max_value=100, value=20)

# Scheduling (optional)
enable_sched = st.sidebar.checkbox("Enable scheduled analysis")
if enable_sched:
    tz = st.sidebar.selectbox("Timezone", pytz.common_timezones, index=pytz.common_timezones.index('US/Pacific'))
    t1 = st.sidebar.time_input("Morning at", datetime.strptime("08:00","%H:%M"))
    t2 = st.sidebar.time_input("Evening at", datetime.strptime("17:00","%H:%M"))

# Configure Gemini API key
if hasattr(st.secrets, 'gemini_api_key'):
    genai.configure(api_key=st.secrets.gemini_api_key)
else:
    st.error("Please set gemini_api_key in .streamlit/secrets.toml")
    st.stop()

# Parsing logic
@st.cache_data(ttl=CACHE_TTL)
def parse_feeds(urls: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=10, headers={'User-Agent':'feed-analyzer'})
            resp.raise_for_status()
            raw = resp.content
        except Exception as e:
            st.warning(f"Failed to fetch {url}: {e}")
            continue
        # RSS/Atom detection
        if b'<rss' in raw[:512] or b'<feed' in raw[:512]:
            feed = feedparser.parse(raw)
            source = feed.feed.get('title', url)
            for entry in feed.entries:
                results.append({
                    'feed_source': source,
                    'title': entry.get('title','').strip(),
                    'description': entry.get('description','').strip(),
                    'published': entry.get('published','').strip()
                })
        else:
            # HTML: try river parser
            items = parse_html_river(raw)
            if items:
                results.extend(items)
            else:
                # Fallback: full text as one item
                text = BeautifulSoup(raw, 'lxml').get_text(separator=' ')
                results.append({
                    'feed_source': url,
                    'title': url,
                    'description': text,
                    'published': ''
                })
    return results


def parse_html_river(html: bytes) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, 'lxml')
    container = soup.find('river') or soup.find(id='river')
    if not container:
        return []
    text = container.get_text(separator='\n')
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    pattern = re.compile(r"(\d{1,2}:\d{2}\s?[AP]M)\s*•\s*([^/]+)/\s*([^:]+):\s*(.*)")
    out = []
    for ln in lines:
        m = pattern.match(ln)
        if not m: continue
        tm, author, src, headline = m.groups()
        out.append({
            'feed_source': src.strip(),
            'title': headline.strip(),
            'description': f"{author.strip()} – {headline.strip()}",
            'published': normalize_time(tm)
        })
    return out


def normalize_time(t: str) -> str:
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        dt = datetime.strptime(f"{today} {t}", "%Y-%m-%d %I:%M %p")
        return dt.isoformat()
    except:
        return t

# Summarization
class APIClient:
    def __init__(self): self.model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
    def get_completion(self, prompt: str) -> str:
        return self.model.generate_content(prompt).text


def summarize(content: str, count: int) -> str:
    prompt = user_prompt.format(feed_count=count)
    full = f"{prompt}\n\n{content}"
    return APIClient().get_completion(full)

# Content extraction
@st.cache_data(ttl=CACHE_TTL)
def extract_content(entries: List[Dict[str, Any]], max_e: int) -> str:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        grouped.setdefault(e['feed_source'], []).append(e)
    parts = []
    for src, items in grouped.items():
        parts.append(f"\nSource: {src}\n" + '-'*40)
        for it in items[:max_e]:
            parts.append(f"Title: {it['title']}\nDate: {it['published']}\nDesc: {it['description']}\n")
    return '\n'.join(parts)

# Main UI
def main():
    if st.button('Run Cross-Feed Analysis'):
        if not urls:
            st.error('Add at least one URL')
            return
        st.write(f"Analyzing {len(urls)} sources...")
        entries = parse_feeds(urls)
        st.write(f"Parsed {len(entries)} entries.")
        content = extract_content(entries, max_entries)
        result = summarize(content, len(urls))
        st.subheader('Analysis')
        st.write(result)
        data = base64.b64encode(result.encode()).decode()
        st.markdown(f"<a href='data:text/plain;base64,{data}' download='analysis.txt'>Download</a>", unsafe_allow_html=True)

# History (optional in sidebar)
def load_history():
    path = 'analysis_history'
    if not os.path.exists(path): return []
    files = sorted(os.listdir(path), reverse=True)
    history = []
    for fn in files[:5]:
        if fn.endswith('.json'):
            with open(f"{path}/{fn}", 'r') as f:
                history.append(json.load(f))
    return history

hist = load_history()
if hist:
    st.sidebar.subheader('History')
    for h in hist:
        st.sidebar.markdown(f"**{h['timestamp']}**: {h['summary'][:80]}...")

# Scheduler
scheduler = BackgroundScheduler(timezone=pytz.UTC)
if enable_sched:
    scheduler.add_job(main, 'cron', hour=t1.hour, minute=t1.minute, timezone=tz)
    scheduler.add_job(main, 'cron', hour=t2.hour, minute=t2.minute, timezone=tz)
    if not scheduler.running: scheduler.start()
else:
    if scheduler.running: scheduler.shutdown()

# Run
if __name__ == '__main__':
    main()
