# 🚀 Streamlit Cloud Deployment Guide

## Quick Deploy (5 Minutes)

### Step 1: Prepare Repository
1. **Fork this repository** to your GitHub account
2. **Ensure** the `streamlit/` folder contains:
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `.streamlit/config.toml` (configuration)

### Step 2: Deploy to Streamlit Cloud
1. **Visit**: [share.streamlit.io](https://share.streamlit.io)
2. **Sign in** with your GitHub account
3. **Click** "New app"
4. **Configure**:
   - Repository: `your-username/Jupiter`
   - Branch: `master` (or `main`)
   - Main file path: `streamlit/app.py`
   - App URL: `jupiter-faq-bot` (or custom name)

### Step 3: Set Environment Variables
In the Streamlit Cloud dashboard:
1. Go to **App Settings** → **Secrets**
2. **Add** the following:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
TOKENIZERS_PARALLELISM = "false"
```

### Step 4: Deploy & Monitor
1. **Click** "Deploy!"
2. **Wait** for initial build (3-5 minutes)
3. **Test** the application
4. **Monitor** logs for any issues

## 🔧 Configuration Details

### Required Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM inference | ✅ Yes |
| `TOKENIZERS_PARALLELISM` | Disable tokenizer warnings | ⚠️ Recommended |

### Optional Configuration
```toml
# Add these to secrets if needed
LOG_LEVEL = "INFO"
CHROMA_PERSIST_DIR = "./data/embeddings/chroma_db"
```

## 📊 Resource Requirements

### Streamlit Cloud Free Tier
- **CPU**: 1 vCPU (sufficient)
- **Memory**: 1 GB RAM (adequate for our models)
- **Storage**: 1 GB (enough for ChromaDB)
- **Bandwidth**: Unlimited

### Performance Expectations
- **Cold Start**: 30-60 seconds first load
- **Warm Response**: 0.6-1.4 seconds
- **Concurrent Users**: 10-50 (free tier)

## 🚨 Troubleshooting

### Common Deployment Issues

**1. Module Import Errors**
```bash
# Solution: Check requirements.txt includes all dependencies
transformers>=4.30.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
groq>=0.4.0
```

**2. Groq API Connection Failed**
```bash
# Check: API key is correct in secrets
# Verify: No extra spaces or quotes in key
```

**3. ChromaDB Initialization Issues**
```bash
# Usually resolves after app restart
# Data persists between deployments
```

**4. Memory Warnings**
```bash
# Add to secrets:
TOKENIZERS_PARALLELISM = "false"
```

### Monitoring & Logs
- **View logs**: Streamlit Cloud dashboard → Logs tab
- **Check health**: Built-in app health indicators
- **Monitor usage**: Analytics dashboard in the app

## 🔄 Updates & Maintenance

### Automatic Updates
- **Git Push**: Automatically triggers redeployment
- **Dependencies**: Updated on each deployment
- **Data**: ChromaDB persists between updates

### Manual Actions
- **Restart App**: Settings → Reboot app
- **Clear Cache**: Restart to clear model cache
- **Update Secrets**: No restart required

## 🔒 Security Best Practices

### API Key Management
- ✅ Store in Streamlit Cloud secrets (encrypted)
- ❌ Never commit API keys to git
- 🔄 Rotate keys periodically

### Access Control
- Public app is read-only
- No user data stored
- Rate limiting built-in

## 📈 Scaling Options

### Free Tier Limits
- 1 app maximum
- Community support
- Public repositories only

### Upgrade Benefits
- Multiple apps
- Private repositories
- Priority support
- Enhanced performance

## 🎯 Production Checklist

- [ ] Repository forked to your GitHub
- [ ] Groq API key obtained and tested
- [ ] Streamlit Cloud account created
- [ ] App deployed successfully
- [ ] Environment variables configured
- [ ] Initial test queries working
- [ ] Analytics dashboard accessible
- [ ] Error handling tested

## 📞 Support

### Getting Help
1. **App Issues**: Check Streamlit Cloud logs
2. **Model Problems**: Verify Groq API status
3. **Data Issues**: Check ChromaDB initialization
4. **General**: Review main project documentation

### Useful Links
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)
- [Groq API Documentation](https://console.groq.com/docs)
- [Project Repository](https://github.com/your-username/Jupiter)

---

**🎉 Once deployed, your Jupiter FAQ Bot will be live at:**
`https://your-app-name.streamlit.app` 