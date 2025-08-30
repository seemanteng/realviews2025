# RealViews Deployment Guide

## Netlify Issue Resolution

The Netlify deployment was failing because:
1. **Wheel vs Source Build**: pip was falling back to building SciPy from source (requiring Fortran compilers)
2. **Python Version Compatibility**: Need Python 3.11 for reliable SciPy wheels
3. **Runtime Requirements**: Streamlit needs a persistent Python server (not static hosting)

### Fix Applied:
- **Python 3.11.9**: Via `runtime.txt` and `PYTHON_VERSION` env var
- **Pinned Wheel Versions**: Exact versions with known wheels in `requirements-netlify.txt`
- **Force Binary**: `PIP_ONLY_BINARY=:all:` prevents source compilation
- **Constraints**: `constraints.txt` ensures consistent versions

## Deployment Solutions

### Option 1: Static Landing Page (Current Netlify Setup)
- **Status**: ✅ Working
- **URL**: Your Netlify URL will show a professional landing page
- **Content**: Project overview, features, and setup instructions
- **Limitation**: No interactive ML functionality

### Option 2: Streamlit Cloud (Recommended for Full App)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Use the original `requirements.txt`
5. Deploy with `app.py` as the main file

### Option 3: Railway
1. Connect your GitHub repo at [railway.app](https://railway.app)
2. Use the original `requirements.txt`
3. Set start command: `streamlit run app.py --server.port=$PORT`
4. Deploy with full ML functionality

### Option 4: Heroku
1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Use the original `requirements.txt`
3. Push to Heroku
4. Set buildpack to Python

### Option 5: Local Development
```bash
git clone https://github.com/seemanteng/realviews
cd realviews
pip install -r requirements.txt
streamlit run app.py
```

## Files Created for Netlify

- `index.html`: Static landing page
- `requirements-netlify.txt`: Lightweight dependencies
- `netlify.toml`: Netlify configuration
- `DEPLOYMENT.md`: This deployment guide

## Performance Comparison

| Platform | Setup Difficulty | ML Features | Cost | Speed |
|----------|-----------------|-------------|------|-------|
| Netlify (Static) | Easy | None | Free | Fast |
| Streamlit Cloud | Easy | Full | Free | Medium |
| Railway | Medium | Full | Free tier + paid | Fast |
| Heroku | Medium | Full | Paid | Medium |
| Local | Easy | Full | Free | Fast |

## Recommendation

For **demo/presentation**: Use Streamlit Cloud or Railway for the full interactive experience.

For **portfolio showcase**: The current Netlify setup provides a professional landing page that loads instantly and showcases your project effectively.