import streamlit as st
import feedparser
import anthropic
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

# LLM Provider Selection
llm_provider = st.sidebar.radio(
    "Select LLM Provider",
    options=["OpenRouter (Free & Paid Models)", "Anthropic (Claude)"],
    help="Choose between OpenRouter's various models (including free options) or Anthropic's Claude"
)

# Model configurations
OPENROUTER_MODELS = {
    "Free Models": {
        "mistralai/mistral-7b-instruct:free": "Mistral 7B (Free)",
        "nousresearch/nous-capybara-7b:free": "Nous Capybara 7B (Free)",
    },
    "Paid Models": {
        "anthropic/claude-3-sonnet": "Claude 3 Sonnet",
        "mistralai/mistral-medium": "Mistral Medium",
        "meta-llama/llama-2-70b-chat": "Llama 2 70B",
    }
}

ANTHROPIC_MODELS = {
    # Current models
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (Latest)",
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku (Latest)",
    "claude-3-opus-20240229": "Claude 3 Opus",
    "claude-3-sonnet-20240229": "Claude 3 Sonnet",
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    # Legacy models
    "claude-2.1": "Claude 2.1 (Legacy)",
    "claude-2.0": "Claude 2.0 (Legacy)",
    "claude-instant-1.2": "Claude Instant 1.2 (Legacy)"
}

# API Key and Model Selection
if llm_provider == "OpenRouter (Free & Paid Models)":
    st.sidebar.subheader("OpenRouter Configuration")
    st.sidebar.markdown("""
    🆓 Free models are available without an API key, but may have:
    - Longer response times
    - Rate limits
    - Lower quality results
    
    For better performance, get an API key at [OpenRouter](https://openrouter.ai/keys)
    """)
    # Use secrets if available, otherwise use text input
    default_openrouter_key = os.getenv('OPENROUTER_API_KEY', '')  # First check environment variable
    if not default_openrouter_key and hasattr(st.secrets, "openrouter_api_key"):
        default_openrouter_key = st.secrets.openrouter_api_key
    openrouter_api_key = st.sidebar.text_input(
        "OpenRouter API Key (optional for free models):",
        value=default_openrouter_key,
        type="password",
        help="Get your API key at https://openrouter.ai/keys"
    )
    
    model_category = st.sidebar.radio("Model Category", ["Free Models", "Paid Models"])
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(OPENROUTER_MODELS[model_category].keys()),
        format_func=lambda x: OPENROUTER_MODELS[model_category][x]
    )
    
    if model_category == "Paid Models" and not openrouter_api_key:
        st.sidebar.warning("⚠️ API key required for paid models")
elif llm_provider == "Anthropic (Claude)":
    st.sidebar.subheader("Anthropic Configuration")
    # Use secrets if available, otherwise use text input
    default_anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')  # First check environment variable
    if not default_anthropic_key and hasattr(st.secrets, "anthropic_api_key"):
        default_anthropic_key = st.secrets.anthropic_api_key
    anthropic_api_key = st.sidebar.text_input(
        "Anthropic API Key:",
        value=default_anthropic_key,
        type="password",
        help="Get your API key at https://console.anthropic.com/settings/keys"
    )
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(ANTHROPIC_MODELS.keys()),
        format_func=lambda x: ANTHROPIC_MODELS[x]
    )

# Default prompt
DEFAULT_PROMPT = """Analyze {feed_count} RSS news feeds and deliver a comprehensive report focusing on talent movement, leadership changes, layoffs, and related announcements. Structure your analysis as follows:

TOP MAJOR TALENT MOVEMENT STORIES:

List up to the top 3 major talent movement stories across all feeds. If more than 3 major stories are identified, list the top 3 and acknowledge the existence of additional significant stories. If fewer than 3 major stories are identified, include only the available ones without fabricating additional stories.

For each major story:
1. Headline & Core Facts
   - Primary Headline/Topic: Clearly state the main headline or topic of the story.
   - Key Facts and Developments: Summarize the essential details, including who is involved (names and titles), what changes are occurring, specific roles or departments affected, and any relevant timelines or numbers.
   - Number of Sources Covering This Story: Indicate how many of the analyzed feeds report on this story.

2. Coverage Analysis
   - Presentation Across Sources: Describe how different sources present the story. Are there variations in emphasis or details?
   - Unique Angles or Perspectives: Highlight any unique viewpoints or additional insights provided by specific sources.
   - Additional Context: Provide any extra background information or context that enhances understanding of the story.

3. Impact & Significance
   - Why This Story Matters: Explain the importance of the story in the context of talent movement and recruitment.
   - Potential Implications: Discuss possible outcomes or effects on the company and its employees, including specific roles or departments that may be impacted.
   - Related Ongoing Developments: Mention any related events or trends that connect to this story.

4. Recruitment Implications
   - Potential Recruitment Opportunities: Identify specific roles, departments, or skill sets that may become available.
   - Talent Pool Insights: Provide insights into the types of candidates who may be seeking new opportunities as a result of this movement.
   - Strategic Recommendations: Offer actionable strategies for recruiters to engage with potential candidates or leverage the movement for talent acquisition.

*If there are more than 3 major stories:*
- **Additional Major Stories:** Note the existence of additional significant talent movement stories without detailing each one.

SECONDARY SIGNIFICANT TALENT MOVEMENT STORIES:

List up to 2-3 other important stories with the following details. Include only those secondary stories that are relevant and do not overlap with major stories.

For each secondary story:
- Headline & Core Facts: Include specific names, companies, roles affected, and key details.
- Coverage: Note which of the analyzed feeds are reporting these stories.
- Coverage Analysis: Briefly describe how the story is presented across sources.
- Impact & Significance: Explain why these stories are significant for talent movement and recruitment strategies.
- Recruitment Implications: Highlight potential recruitment opportunities or talent pool changes arising from these stories.

SOURCE COMPARISON:

- Comprehensive Coverage: Identify which sources provide the most detailed and extensive coverage of talent movement stories.
- Differences in Story Selection: Highlight any significant variations in the types of stories different sources prioritize.
- Unique Stories: Point out any unique stories that are exclusively covered by single sources.
- Source Reliability: Assess the reliability and focus areas of each source in relation to talent movement insights.

NOTABLE QUOTES:

- Significant Quotes: Include 2-3 direct quotes from sources that are particularly insightful or impactful regarding talent movement.
- Context for Each Quote: Provide background or explanation for each quote to understand its relevance.
- Source Attribution and Significance: Note which source provided the quote and why it holds importance.

GUIDELINES:

- Specificity: Focus on concrete details, including names, numbers, dates, and factual information rather than vague statements.
- Relevance to Recruitment: Emphasize aspects of each story that directly impact talent acquisition and recruitment strategies.
- Clarity and Organization: Present information in a clear, well-organized manner to facilitate quick understanding and decision-making.
- Actionable Insights: Where possible, highlight insights that can inform proactive recruitment actions."""

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
    def __init__(self, api_key: str, provider: str):
        self.api_key = api_key
        self.provider = provider
        if provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
    
    def get_completion(self, content: str, model: str) -> str:
        if not self.api_key:
            raise ValueError("API key is required")
            
        for attempt in range(MAX_RETRIES):
            try:
                if self.provider == "openrouter":
                    return self._get_openrouter_completion(content, model)
                else:  # anthropic
                    return self._get_anthropic_completion(content, model)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise e
                sleep(RETRY_DELAY)
    
    def _get_openrouter_completion(self, content: str, model: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://rss-analyzer-mock.streamlit.app",  # Update this
            "X-Title": "RSS Feed Analyzer",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert news analyst and pattern recognition specialist."
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.1,
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=180  # Increased timeout
            )
            
            response_json = response.json()
            
            # Show response in expander for debugging
            with st.expander("Debug: API Response", expanded=False):
                st.json(response_json)
            
            if "error" in response_json:
                error_msg = response_json["error"].get("message", "Unknown error")
                raise Exception(f"OpenRouter API Error: {error_msg}")
            
            if "choices" in response_json and len(response_json["choices"]) > 0:
                return response_json["choices"][0]["message"]["content"]
            else:
                raise Exception("No valid response content found")
                
        except Exception as e:
            st.error(f"Error with model {model}: {str(e)}")
            if not self.api_key and "authentication" in str(e).lower():
                st.warning("⚠️ You may need an API key for better performance. Get one at https://openrouter.ai/keys")
            raise e

    def _get_anthropic_completion(self, content: str, model: str) -> str:
        message = self.client.messages.create(
            model=model,
            max_tokens=1500,
            temperature=0,
            system="You are an expert news analyst and pattern recognition specialist tasked with analyzing multiple RSS feeds to identify themes, patterns, and significant stories across different sources.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        }
                    ]
                }
            ]
        )
        return message.content[0].text

def initialize_session_state():
    # Initialize provider selection if not already set
    if 'llm_provider' not in st.session_state:
        st.session_state['llm_provider'] = "OpenRouter (Free & Paid Models)"
    
    # Only create API client when needed (during analysis)
    if 'api_client' not in st.session_state and st.session_state.get('_run_analysis', False):
        api_key = ''
        provider = st.session_state.get('llm_provider')
        
        if provider == "OpenRouter (Free & Paid Models)":
            api_key = st.session_state.get('openrouter_api_key', '')
            if not api_key and st.session_state.get('selected_model') in OPENROUTER_MODELS["Paid Models"]:
                raise ValueError("OpenRouter API key is required for paid models")
        elif provider == "Anthropic (Claude)":
            api_key = st.session_state.get('anthropic_api_key', '')
            if not api_key:
                raise ValueError("Anthropic API key is required")
        
        # Verify API key is not empty
        if not api_key:
            raise ValueError(f"API key is required for {provider}")
            
        st.session_state.api_client = APIClient(
            api_key=api_key,
            provider='openrouter' if provider == "OpenRouter (Free & Paid Models)"
                    else 'anthropic'
        )

# Functions
def generate_summary(content, feed_count, prompt_template):
    formatted_prompt = prompt_template.format(feed_count=feed_count)
    try:
        initialize_session_state()
        provider = st.session_state.get('llm_provider')
        
        if provider == "OpenRouter (Free & Paid Models)":
            model_name = st.session_state.get('selected_model')
            model_category = st.session_state.get('model_category', 'Free Models')
            if not model_name or model_name not in OPENROUTER_MODELS[model_category]:
                st.error(f"Please select a valid model from {model_category}")
                raise ValueError("No valid model selected")
        else:  # Anthropic (Claude)
            model_name = st.session_state.get('selected_model')
            if not model_name or model_name not in ANTHROPIC_MODELS:
                st.error("Please select a valid Anthropic model")
                raise ValueError("No valid model selected")
            
        if not model_name:
            raise ValueError("No model selected")
        
        # Ensure content is properly encoded
        cleaned_content = _clean_text(content)
        
        summary = st.session_state.api_client.get_completion(
            content=f"{formatted_prompt}\n\nContent to analyze:\n{cleaned_content}",
            model=model_name
        )
        
        # Save the summary
        timestamp = datetime.now()
        os.makedirs('analysis_history', exist_ok=True)
        filename = f'analysis_history/analysis_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp.isoformat(),
                'summary': summary,
                'model_used': model_name
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
    # Store API keys and provider in session state
    st.session_state['llm_provider'] = llm_provider
    if llm_provider == "OpenRouter (Free & Paid Models)":
        st.session_state['openrouter_api_key'] = openrouter_api_key
        st.session_state['selected_model'] = selected_model
        st.session_state['model_category'] = model_category
    else:
        st.session_state['anthropic_api_key'] = anthropic_api_key
        st.session_state['selected_model'] = selected_model
        st.session_state['model_category'] = None

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

    tab1, tab2 = st.tabs(["Current Analysis", "Analysis History"])
    
    with tab1:
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
                
                summary = generate_summary(content, feed_count, user_prompt)
                st.subheader("Cross-Feed Analysis:")
                st.write(summary)
                
                timestamp = datetime.now()
                save_analysis_result(summary, timestamp)
                st.caption(f"Analysis generated at: {timestamp.strftime('%Y%m-%d %H:%M:%S')} using {selected_model}")
                
            except ValueError as ve:
                st.error(f"Configuration error: {str(ve)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                st.session_state['_run_analysis'] = False
    
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
