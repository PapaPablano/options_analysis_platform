# 🚀 Deployment Guide - Stock Market Trends Platform

## 📋 Prerequisites

### System Requirements
- Python 3.9+ 
- 4GB+ RAM (8GB recommended for LSTM models)
- 2GB+ free disk space
- Internet connection for data fetching

### Required Software
- Python 3.9 or higher
- pip (Python package manager)
- Git (for version control)

## 🛠️ Local Deployment

### Step 1: Environment Setup
```bash
# Clone or navigate to your project directory
cd /path/to/options_analysis_platform

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Optional - Fix XGBoost (macOS)
```bash
# Install OpenMP for XGBoost (if you have Homebrew)
brew install libomp

# Alternative: Install XGBoost without OpenMP
pip uninstall xgboost
pip install xgboost --no-deps
```

### Step 3: Run the Applications
```bash
# Options Analysis Dashboard
python3 -m streamlit run simple_dashboard.py --server.port 8501

# Stock Market Trends Dashboard (in separate terminal)
python3 -m streamlit run trends_dashboard.py --server.port 8502
```

### Step 4: Access Your Platform
- **Options Dashboard**: http://localhost:8501
- **Trends Dashboard**: http://localhost:8502

## ☁️ Cloud Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free)

#### Prerequisites
- GitHub account
- Your code in a GitHub repository

#### Steps
1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/your-repo.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Choose main file: `simple_dashboard.py` or `trends_dashboard.py`
   - Click "Deploy"

3. **Configure for Multiple Apps**:
   - Deploy `simple_dashboard.py` first
   - Deploy `trends_dashboard.py` as second app
   - Each gets its own URL

#### Streamlit Cloud Configuration
Create `.streamlit/config.toml`:
```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

### Option 2: Heroku Deployment

#### Prerequisites
- Heroku account
- Heroku CLI installed

#### Steps
1. **Create Procfile**:
   ```
   web: streamlit run simple_dashboard.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create runtime.txt**:
   ```
   python-3.9.18
   ```

3. **Deploy**:
   ```bash
   heroku create your-app-name
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### Option 3: Docker Deployment

#### Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "simple_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run
```bash
# Build Docker image
docker build -t stock-trends-platform .

# Run container
docker run -p 8501:8501 stock-trends-platform
```

### Option 4: AWS/GCP Deployment

#### AWS EC2
1. Launch EC2 instance (t3.medium or larger)
2. Install Docker or Python directly
3. Clone repository
4. Install dependencies
5. Run with systemd service

#### Google Cloud Run
1. Build Docker image
2. Push to Google Container Registry
3. Deploy to Cloud Run
4. Configure custom domain (optional)

## 🔧 Production Configuration

### Environment Variables
Create `.env` file:
```bash
# API Keys (if needed)
YAHOO_FINANCE_API_KEY=your_key_here

# Model Configuration
MODEL_CACHE_DIR=/tmp/models
MAX_WORKERS=4

# Security
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
```

### Performance Optimization

#### For Large Datasets
```python
# In your Streamlit app, add caching
@st.cache_data
def load_data(ticker):
    # Your data loading code
    pass

@st.cache_resource
def load_model():
    # Your model loading code
    pass
```

#### Memory Management
```python
# Add to your dashboard
import gc
import psutil

# Monitor memory usage
if psutil.virtual_memory().percent > 80:
    gc.collect()
    st.warning("High memory usage detected. Clearing cache.")
```

### Security Considerations

#### Streamlit Configuration
```toml
# .streamlit/config.toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

#### API Rate Limiting
```python
# Add rate limiting to your data fetching
import time
from functools import wraps

def rate_limit(calls_per_minute=60):
    def decorator(func):
        last_called = [0.0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = 60.0 / calls_per_minute - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator
```

## 📊 Monitoring and Maintenance

### Health Checks
Create `health_check.py`:
```python
import requests
import sys

def check_app_health(url):
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    if check_app_health("http://localhost:8501"):
        print("✅ App is healthy")
        sys.exit(0)
    else:
        print("❌ App is down")
        sys.exit(1)
```

### Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Backup Strategy
```bash
# Backup models and data
tar -czf backup_$(date +%Y%m%d).tar.gz models/ data/ logs/
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port
lsof -i :8501
# Kill process
kill -9 <PID>
```

#### 2. Memory Issues
```bash
# Monitor memory usage
htop
# Or use Python
python3 -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

#### 3. XGBoost Issues
```bash
# Reinstall XGBoost
pip uninstall xgboost
pip install xgboost
```

#### 4. TensorFlow Issues
```bash
# Clear TensorFlow cache
rm -rf ~/.cache/tensorflow
```

### Performance Issues
- Reduce model complexity for faster training
- Use smaller datasets for testing
- Implement model caching
- Use GPU acceleration if available

## 📈 Scaling Considerations

### For High Traffic
- Use load balancer (nginx)
- Implement Redis caching
- Use database for model storage
- Consider microservices architecture

### For Large Datasets
- Implement data streaming
- Use distributed computing (Dask, Ray)
- Optimize feature engineering
- Use incremental learning

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
# .github/workflows/deploy.yml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Streamlit Cloud
      run: |
        # Your deployment commands here
        echo "Deploying to Streamlit Cloud..."
```

## 📞 Support and Maintenance

### Regular Tasks
- Update dependencies monthly
- Monitor performance metrics
- Backup models and data
- Review and update documentation

### Emergency Procedures
- Keep backup deployment ready
- Monitor error logs
- Have rollback plan
- Document incident response

---

## 🎯 Quick Start Commands

```bash
# Local development
python3 -m streamlit run simple_dashboard.py

# Production deployment (Streamlit Cloud)
git push origin main  # After setting up GitHub repo

# Docker deployment
docker build -t stock-trends . && docker run -p 8501:8501 stock-trends

# Health check
python3 health_check.py
```

Your platform is now ready for production deployment! 🚀
