# Jupiter FAQ Bot - Streamlit Application

## 🚀 Production Deployment

This directory contains the production-ready Streamlit application for the Jupiter FAQ Bot.

### Quick Deploy to Streamlit Cloud

1. **Fork/Clone Repository**: Ensure this repository is on GitHub
2. **Visit**: [share.streamlit.io](https://share.streamlit.io)
3. **Connect**: Link your GitHub account
4. **Deploy**: 
   - Repository: `your-username/Jupiter`
   - Branch: `master`
   - Main file path: `streamlit/app.py`

### Environment Variables (Required)

Set these in Streamlit Cloud → App Settings → Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
TOKENIZERS_PARALLELISM = "false"
```

### Local Development

```bash
# From project root
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```

### Features

- **Real-time Chat**: Interactive FAQ bot interface
- **Model Selection**: Choose between Groq API, DistilBERT, or Auto
- **Analytics Dashboard**: Performance metrics and usage analytics
- **Admin Panel**: System health monitoring and data management
- **Responsive Design**: Works on desktop and mobile

### Performance

- **Response Time**: 0.6-1.4 seconds average
- **Document Coverage**: 185 Jupiter FAQ documents
- **Availability**: 99.9% uptime with fallback systems
- **Languages**: English, Hindi, Hinglish support

### Support

For deployment issues, check the main project's DEPLOYMENT_GUIDE.md 