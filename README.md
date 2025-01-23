# RSS Feed Analyzer

AI-powered RSS feed analyzer with chat capabilities using Google's Gemini model.

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your API keys:
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Add your API keys in the secrets.toml file
   
## Usage

Run the app locally:
```bash
streamlit run app.py
```

## Configuration

- Supports both OpenRouter and Anthropic Claude models
- Customizable analysis prompts
- Multiple RSS feed sources
- Scheduled analysis options

## Deployment

To deploy on Streamlit Community Cloud:

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy the app
5. Add your API keys in the Streamlit Cloud secrets management
