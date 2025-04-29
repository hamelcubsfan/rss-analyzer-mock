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
Review the latest news from {feed_count} sources and produce a recruiter-focused report with a consistent structure, capturing all high-impact disruptions, movements, or must-know stories published today (or within the last hour if available). The report should educate the team on critical industry developments, market updates, employment changes, and recruitment opportunities. Cross-reference stories across sources to highlight connections and trends. Focus on:

- Executive moves: promotions, departures, leadership changes.
- Employment updates: layoffs, hiring freezes, major restructuring, or expansions.
- Major industry news: significant market shifts, technological breakthroughs, regulatory changes, funding rounds, or partnerships.
- Recruitment opportunities: in-demand roles, skills, and sourcing recommendations.

### Instructions
- Identify and prioritize stories with significant industry or recruiting impact (e.g., large-scale layoffs, major funding, C-suite changes, disruptive innovations).
- Cross-reference related stories across sources to provide a cohesive narrative (e.g., link a company's layoff announcement to its funding news or market shift).
- For each entry, extract actionable recruiting insights where applicable (e.g., skills in demand, talent pools to target) and highlight broader implications for team awareness.
- Ensure no duplicate stories; merge details from multiple sources into a single entry if they cover the same event.
- Include all relevant, high-impact stories published today, prioritizing those from the last hour if timestamps allow.
- If a section has many stories, prioritize the most impactful but include all that meet the "must-know" threshold for team education (e.g., major disruptions, opportunities, or market updates).

### Output Structure
1) **Overview**: 3-5 bullet points summarizing key highlights, prioritizing high-impact disruptions, market updates, or trends across sources to inform the team.
2) **Executive Moves**:
   - *Title*: Brief headline (e.g., "CTO Departs TechCorp").
   - *Source(s)*: List all sources reporting the story.
   - *Time*: Publication date/time (include hour if available).
   - *Key Details*: Summary of the move and its context.
   - *Team Impact*: Implications for recruiting or industry awareness (e.g., opportunities to recruit from affected teams).
3) **Employment Updates**:
   - Same structure as above, covering layoffs, hiring freezes, restructuring, or expansions.
4) **Major Industry News**:
   - Same structure, covering market shifts, technological breakthroughs, regulatory changes, funding rounds, or partnerships.
5) **Recruitment Opportunities**:
   - *Title*: Role or skill area (e.g., "AI Engineers in Demand").
   - *Source(s)*: Sources indicating demand.
   - *Key Details*: Why the role/skill is trending, industries affected.
   - *Sourcing Recommendations*: Specific strategies (e.g., target LinkedIn groups, universities, or competitors).

### Constraints
- Include all high-impact, must-know stories per section from today, typically aiming for 2-5 entries but allowing more if necessary to cover critical updates for team education.
- Use clear, concise language (50-100 words per entry).
- If no relevant stories exist for a section, state "No significant updates today" and explain briefly.
"""
user_prompt = st.sidebar.text_area("Analysis prompt:", value=DEFAULT_PROMPT, height=200)
if st.sidebar.button("Reset prompt to default"):
    user_prompt = DEFAULT_PROMPT
    st.sidebar.success("Prompt reset")

# URL input
urls_text = st.sidebar.text_area(
    "Enter RSS or HTML URLs (one per line):",
    value="https://techcrunch.com/feed/\nhttps://www.techmeme.com/river\nhttps://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910\nhttps://www.theverge.com/rss/index.xml\nhttps://gizmodo.com/feed",
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
            resp = requests.get(url, timeout=10)
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
            # HTML page: extract main content via <article> or <main>
            soup = BeautifulSoup(raw, 'lxml')
            container = soup.find('article') or soup.find('main') or soup.body or soup
            # Gather paragraphs, skip tiny/boilerplate ones
            paragraphs = container.find_all('p')
            text_lines = []
            for p in paragraphs:
                t = p.get_text().strip()
                if len(t) < 50:
                    continue
                # skip common boilerplate
                if any(kw in t for kw in ("©","Subscribe","Follow","Advertisement")):
                    continue
                text_lines.append(t)
            text = "\n\n".join(text_lines)
            # fallback if nothing extracted
            if not text:
                text = container.get_text(separator=' ', strip=True)
            # extract title
            title_tag = soup.find('h1') or soup.find('title')
            title = title_tag.get_text().strip() if title_tag else url
            results.append({
                'feed_source': url,
                'title': title,
                'description': text,
                'published': ''
            })
    return results

# Summarization
class APIClient:
    def __init__(self):
        self.model = genai.GenerativeModel('models/gemini-2.5-flash-preview-04-17')
    def get_completion(self, prompt: str) -> str:
        return self.model.generate_content(prompt).text


def summarize(content: str, count: int) -> str:
    max_len = 15000
    truncated = False
    if len(content) > max_len:
        content = content[:max_len]
        truncated = True
    prompt_text = user_prompt.format(feed_count=count)
    full_prompt = f"{prompt_text}\n\n{content}"
    try:
        response = APIClient().get_completion(full_prompt)
    except Exception:
        fallback = content[:5000]
        full_prompt = f"{prompt_text}\n\n{fallback}"
        response = APIClient().get_completion(full_prompt)
    if truncated:
        response += "\n[Output truncated]"
    return response

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

# Run
if __name__ == '__main__':
    main()
