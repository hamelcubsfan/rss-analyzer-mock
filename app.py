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
import google.generativeai as genai  # Add this import at the top
import base64
import tempfile
from fpdf import FPDF

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
st.title("📰 RSS Feed Analyzer")
st.write("Discover insights across multiple RSS feeds using AI-powered analysis")

# Sidebar title
st.sidebar.header("⚙️ Configuration")

# Set up Gemini API key from secrets
if hasattr(st.secrets, "gemini_api_key"):
    genai.configure(api_key=st.secrets.gemini_api_key)
else:
    st.error("Gemini API key not found in secrets. Please add it to your secrets.toml file.")
    st.stop()

# Simplify model selection - only show Gemini
st.sidebar.markdown("""
🤖 Using Google's Gemini 2.0 Flash model
- Provides fast, high-quality responses
- Optimized for analysis tasks
""")

# Remove the old model selection code and replace with:
selected_model = "gemini-2.0-flash-thinking-exp"

# Default prompt
DEFAULT_PROMPT = """Analyze {feed_count} RSS news feeds and produce a well-structured report on talent movement, leadership changes, layoffs, and related announcements. Ensure the final output is clearly organized, with consistent numbering, bullet points, and spacing.

TOP MAJOR TALENT MOVEMENT STORIES:
- Identify up to the top 3 major talent movement stories across all feeds.
- If more than 3 major stories are found, list the top 3 and note that additional significant stories exist.
- If fewer than 3 major stories are identified, list all that meet the criteria.

For each major story:
1. Headline & Core Facts
   - Primary Headline/Topic: Clearly state the main headline or topic.
   - Key Facts and Developments: Summarize essential details (names, titles, changes, roles/departments affected, timelines).
   - Number of Sources Covering This Story: Indicate how many feeds report on this story.

2. Coverage Analysis
   - Presentation Across Sources: Describe variations in emphasis or details across sources.
   - Unique Angles or Perspectives: Highlight additional insights or viewpoints from specific sources.
   - Additional Context: Provide any background details that enrich understanding of the story.

3. Impact & Significance
   - Why This Story Matters: Explain its importance in relation to talent movement and recruitment.
   - Potential Implications: Discuss possible outcomes or effects on the company and its employees (roles or departments potentially impacted).
   - Related Ongoing Developments: Mention any related events or trends connected to this story.

4. Recruitment Implications
   - Potential Recruitment Opportunities: Identify roles, departments, or skill sets that may become available.
   - Talent Pool Insights: Note the types of candidates who may seek new opportunities due to this movement.
   - Strategic Recommendations: Provide actionable strategies for recruiters to engage or leverage these developments.

If there are more than 3 major stories:
- Additional Major Stories: Briefly acknowledge other significant talent movement stories without detailing each one.

SECONDARY SIGNIFICANT TALENT MOVEMENT STORIES:
- List 2–3 other important stories that are relevant but do not overlap with major stories.

For each secondary story:
- Headline & Core Facts: Include names, companies, roles, and key details.
- Coverage: Note which feeds report the story.
- Coverage Analysis: Briefly describe any variations across sources.
- Impact & Significance: Explain the relevance for talent movement and recruitment strategies.
- Recruitment Implications: Highlight possible recruitment opportunities or shifts in the talent pool.

SOURCE COMPARISON:
- Comprehensive Coverage: Identify sources that provide the most detailed reporting on talent movement.
- Differences in Story Selection: Highlight significant variations in story priorities among sources.
- Unique Stories: Point out any exclusive stories covered by only one source.
- Source Reliability: Offer a brief assessment of each source’s reliability and focus regarding talent movement.

NOTABLE QUOTES:
- Significant Quotes: Include 2–3 direct quotes that offer insightful or impactful information about talent movement.
- Context for Each Quote: Provide background or explanation for clarity.
- Source Attribution and Significance: Mention the source and why the quote is valuable.

GUIDELINES:
- Specificity: Use clear data points (names, numbers, dates) instead of vague statements.
- Relevance to Recruitment: Emphasize how each story affects talent acquisition and recruitment strategies.
- Clarity and Organization: Present the information in an orderly, easy-to-read format.
- Actionable Insights: Highlight advice or recommendations that inform proactive recruitment actions.

OUTPUT FORMAT REQUIREMENTS:
1. Use consistent numbering and bullet points.
2. Maintain clear headings and subheadings.
3. Avoid extra line breaks or inconsistent spacing.
4. Present the final report in a clean, readable format."""

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

@st.cache_data(ttl=CACHE_TTL)
def parse_feeds(rss_feeds: List[str]) -> List[Dict[str, Any]]:
    all_entries = []
    
    for feed_url in rss_feeds:
        for attempt in range(MAX_RETRIES):
            try:
                with st.spinner(f"Parsing feed: {feed_url}"):
                    feed = feedparser.parse(feed_url)
                    feed_title = feed.feed.get('title', feed_url)
                    
                    # More robust entry checking
                    if not hasattr(feed, 'entries') or not feed.entries:
                        st.warning(f"No entries found in feed: {feed_url}")
                        break
                    
                    # Clean and sanitize entries
                    entries = []
                    for entry in feed.entries:
                        cleaned_entry = {
                            'feed_source': feed_title,
                            'title': _clean_text(entry.get('title', 'No title')),
                            'description': _clean_text(entry.get('description', 'No description available')),
                            'published': entry.get('published', 'No date available')
                        }
                        entries.append(cleaned_entry)
                    
                    all_entries.extend(entries)
                    break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    st.error(f"Failed to parse {feed_url}: {str(e)}")
                else:
                    sleep(RETRY_DELAY)
    
    return all_entries

def _clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not isinstance(text, str):
        return str(text)
    # Replace problematic characters and normalize
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    return text.encode('utf-8', 'ignore').decode('utf-8')

@st.cache_data(ttl=CACHE_TTL)
def extract_content(entries: List[Dict[str, Any]], max_entries: int = 10) -> str:
    grouped_entries = {}
    
    for entry in entries:
        source = entry.get('feed_source', 'Unknown Source')
        if source not in grouped_entries:
            grouped_entries[source] = []
        if len(grouped_entries[source]) < max_entries:
            grouped_entries[source].append(entry)
    
    return "\n".join(
        f"\nFeed Source: {source}\n{'='*50}\n" +
        "\n".join(
            f"Title: {entry.get('title', 'No title')}\n"
            f"Date: {entry.get('published', 'No date available')}\n"
            f"Description: {entry.get('description', 'No description available')}\n"
            for entry in source_entries
        )
        for source, source_entries in grouped_entries.items()
    )

class APIClient:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')
    
    def get_completion(self, content: str, model: str) -> str:
        try:
            response = self.model.generate_content(content)
            return response.text
        except Exception as e:
            raise Exception(f"Error with Gemini model: {str(e)}")

def initialize_session_state():
    # Simplify session state initialization
    if 'api_client' not in st.session_state and st.session_state.get('_run_analysis', False):
        st.session_state.api_client = APIClient()

def generate_summary(content, feed_count, prompt_template):
    formatted_prompt = prompt_template.format(feed_count=feed_count)
    try:
        initialize_session_state()
        
        # Ensure content is properly encoded
        cleaned_content = _clean_text(content)
        
        summary = st.session_state.api_client.get_completion(
            content=f"{formatted_prompt}\n\nContent to analyze:\n{cleaned_content}",
            model="gemini-2.0-flash-thinking-exp"  # Use the model directly
        )
        
        # Save the summary
        timestamp = datetime.now()
        os.makedirs('analysis_history', exist_ok=True)
        filename = f'analysis_history/analysis_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp.isoformat(),
                'summary': summary,
                'model_used': selected_model
            }, f, ensure_ascii=False)
            
        return summary
    except Exception as e:
        raise Exception(f"Error generating summary: {str(e)}")

def load_analysis_history():
    if not os.path.exists('analysis_history'):
        return []
    
    history = []
    for filename in sorted(os.listdir('analysis_history'), reverse=True):
        if filename.endswith('.json'):
            with open(f'analysis_history/{filename}', 'r') as f:
                history.append(json.load(f))
    return history

def run_scheduled_analysis():
    try:
        entries = parse_feeds(rss_feeds)
        content = extract_content(entries, max_entries)
        summary = generate_summary(content, len(rss_feeds), user_prompt)
        save_analysis_result(summary, datetime.now())
    except Exception as e:
        print(f"Scheduled analysis error: {str(e)}")

def save_analysis_result(summary: str, timestamp: datetime) -> None:
    """Save analysis result to a JSON file in the analysis_history directory."""
    os.makedirs('analysis_history', exist_ok=True)
    filename = f'analysis_history/analysis_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
    
    analysis_data = {
        'timestamp': timestamp.isoformat(),
        'summary': summary,
        'model_used': st.session_state.get('selected_model', 'unknown')
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)

# Initialize scheduler
scheduler = BackgroundScheduler(timezone=pytz.UTC)

# Main app logic
def main():
    global rss_feeds, max_entries  # Add this line to fix the reference error
    
    # Remove provider-specific initialization
    st.session_state['selected_model'] = "gemini-2.0-flash-thinking-exp"

    # RSS Feeds input
    st.header("RSS Feed Configuration")
    st.write("Enter your RSS feed URLs below (5-10 recommended for optimal analysis)")
    
    default_feeds = """https://techcrunch.com/feed/
https://www.techmeme.com/feed.xml
https://mashable.com/feeds/rss/all
https://www.geekwire.com/feed/
https://www.wired.com/feed/tag/ai/latest/rss"""
    
    rss_feeds = st.text_area(
        "Enter RSS feed URLs (one per line):", 
        value=default_feeds,
        height=200,
        help="Add 5-10 RSS feeds for best cross-feed analysis results"
    )
    rss_feeds = [url.strip() for url in rss_feeds.split("\n") if url.strip()]
    
    feed_count = len(rss_feeds)
    st.write(f"Number of feeds to analyze: {feed_count}")
    if feed_count < 5:
        st.warning("⚠️ Adding more feeds (minimum 5 recommended) will improve cross-feed analysis")
    
    max_entries = st.number_input(
        "Number of entries to analyze per feed:", 
        min_value=1, 
        max_value=50, 
        value=10,
        help="Higher values provide more comprehensive analysis but take longer to process"
    )

    # Change tabs definition from 3 to 2
    tab1, tab2 = st.tabs(["Current Analysis", "Analysis History"])
    
    with tab1:
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None

        if st.button("Run Cross-Feed Analysis"):
            try:
                st.session_state['_run_analysis'] = True
                initialize_session_state()
                
                entries = parse_feeds(rss_feeds)
                st.write(f"Total entries parsed: {len(entries)}")
                
                if not entries:
                    st.error("No entries were parsed from any feeds. Please check your feed URLs.")
                    return
                    
                content = extract_content(entries, max_entries)
                st.write(f"Content extracted from {feed_count} feeds successfully.")
                
                timestamp = datetime.now()
                summary = generate_summary(content, feed_count, user_prompt)
                st.session_state.current_analysis = summary  # Store in session state
                st.session_state.current_timestamp = timestamp  # Store timestamp in session state
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                st.session_state['_run_analysis'] = False

        # Only show analysis once with download options
        if st.session_state.current_analysis:
            st.subheader("Cross-Feed Analysis:")
            st.write(st.session_state.current_analysis)
            
            timestamp = getattr(st.session_state, 'current_timestamp', datetime.now())
            
            # Add download buttons
            st.write("📥 Download Analysis:")
            download_col1, download_col2, download_col3 = st.columns([1, 1, 2])
            
            with download_col1:
                txt_content = f"Analysis Generated at: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n{st.session_state.current_analysis}"
                b64_txt = base64.b64encode(txt_content.encode()).decode()
                href_txt = f'<a href="data:text/plain;base64,{b64_txt}" download="analysis_{timestamp.strftime("%Y%m%d_%H%M%S")}.txt">Download as TXT</a>'
                st.markdown(href_txt, unsafe_allow_html=True)
            
            with download_col2:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, txt=txt_content)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    pdf.output(tmp_file.name)
                    with open(tmp_file.name, "rb") as f:
                        b64_pdf = base64.b64encode(f.read()).decode()
                    os.unlink(tmp_file.name)
                
                href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="analysis_{timestamp.strftime("%Y%m%d_%H%M%S")}.pdf">Download as PDF</a>'
                st.markdown(href_pdf, unsafe_allow_html=True)

            st.caption(f"Analysis generated using {selected_model}")
            save_analysis_result(st.session_state.current_analysis, timestamp)

        # Chat interface
        st.markdown("---")
        st.subheader("💬 Chat with Analysis")
        
        # Initialize API client for chat
        if 'api_client' not in st.session_state:
            st.session_state.api_client = APIClient()
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Get the latest analysis from session state or history
        analysis_to_use = st.session_state.current_analysis
        if not analysis_to_use:
            history = load_analysis_history()
            if history:
                analysis_to_use = history[0]['summary']
        
        if not analysis_to_use:
            st.info("Run an analysis above to start chatting about the results.")
        else:
            chat_col1, chat_col2 = st.columns([3, 1])
            
            with chat_col1:
                user_question = st.text_input("Ask a question about the analysis:", 
                    key="chat_input", 
                    placeholder="e.g., 'What are the top recruitment opportunities?'")
            
            with chat_col2:
                ask_button = st.button("Ask", key="ask_button", use_container_width=True)
            
            if ask_button and user_question:
                try:
                    chat_prompt = f"""Based on the following analysis, please answer the question: "{user_question}"
                    
Analysis:
{analysis_to_use}

Please provide a clear and concise response."""
                    
                    response = st.session_state.api_client.get_completion(
                        content=chat_prompt,
                        model=selected_model
                    )
                    
                    # Update chat history without clearing the page
                    st.session_state.chat_history.append(("You", user_question))
                    st.session_state.chat_history.append(("Assistant", response))
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

            # Display chat history
            st.markdown("#### Recent Conversation")
            chat_container = st.container()
            with chat_container:
                for role, message in reversed(st.session_state.chat_history[-6:]):
                    if role == "You":
                        st.markdown(f"**👤 You:** {message}")
                    else:
                        st.markdown(f"**🤖 Assistant:** {message}")
                    st.markdown("---")

    with tab2:
        st.subheader("Previous Analyses")
        history = load_analysis_history()
        if not history:
            st.info("No previous analyses found.")
        else:
            for analysis in history:
                with st.expander(f"Analysis from {analysis['timestamp']} using {analysis.get('model_used', 'unknown model')}"):
                    st.write(analysis['summary'])

    if enable_scheduling:
        scheduler.remove_all_jobs()
        scheduler.add_job(
            run_scheduled_analysis,
            'cron',
            hour=morning_time.hour,
            minute=morning_time.minute,
            timezone=schedule_timezone
        )
        scheduler.add_job(
            run_scheduled_analysis,
            'cron',
            hour=evening_time.hour,
            minute=evening_time.minute,
            timezone=schedule_timezone
        )
        if not scheduler.running:
            scheduler.start()
        st.sidebar.success(f"Scheduled analyses set for {morning_time.strftime('%H:%M')} and {evening_time.strftime('%H:%M')} {schedule_timezone}")
    else:
        if scheduler.running:
            scheduler.shutdown()

if __name__ == "__main__":
    initialize_session_state()
    main()
