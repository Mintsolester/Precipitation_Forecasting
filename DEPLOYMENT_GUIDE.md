# Deploying the Precipitation Forecasting App

## 📋 Pre-Deployment Checklist

### Required Files (Already Created):
- ✅ `app.py` - Main Streamlit application
- ✅ `requirements_streamlit.txt` - Python dependencies
- ✅ `packages.txt` - System dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.gitignore` - Files to exclude from Git
- ✅ Model files in `runs/lstm_flood_prediction/`
- ✅ Data file in `data/processed/era5/ganges_farakka_era5_daily.csv`

### Important Files to Include:
```
neuralhydrology-flood-forecast/
├── app.py
├── requirements_streamlit.txt
├── packages.txt
├── .streamlit/
│   └── config.toml
├── scripts/
│   └── train_lstm_flood.py  (needed for model class)
├── runs/
│   └── lstm_flood_prediction/
│       ├── best_model.pt  (~839 KB)
│       └── scaler.pkl  (~570 B)
└── data/
    └── processed/
        └── era5/
            └── ganges_farakka_era5_daily.csv  (~200 KB)
```

---

## 🚀 Option 1: Streamlit Community Cloud (RECOMMENDED - FREE)

### Step 1: Prepare GitHub Repository

```bash
cd /home/aryan/neuralhydrology-flood-forecast

# Initialize git if not already done
git init

# Add files
git add app.py
git add requirements_streamlit.txt
git add packages.txt
git add .streamlit/
git add scripts/train_lstm_flood.py
git add runs/lstm_flood_prediction/
git add data/processed/era5/ganges_farakka_era5_daily.csv

# Commit
git commit -m "Add Streamlit precipitation forecasting app"

# Create repository on GitHub and push
git remote add origin https://github.com/YOUR_USERNAME/neuralhydrology-flood-forecast.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - **Repository**: `YOUR_USERNAME/neuralhydrology-flood-forecast`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click "Deploy!"

**Deployment time**: ~5-10 minutes

**Your app URL**: `https://YOUR_USERNAME-neuralhydrology-flood-forecast.streamlit.app`

### Pros:
- ✅ **FREE** (unlimited public apps)
- ✅ Auto-deploys on git push
- ✅ HTTPS by default
- ✅ No server management

### Cons:
- ❌ Must be public (or pay for private)
- ❌ Limited resources (1 GB RAM, 1 CPU)
- ❌ May sleep after inactivity

---

## 🐳 Option 2: Docker Deployment (Local or Cloud)

### Create Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libhdf5-dev \\
    libnetcdf-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_streamlit.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_streamlit.txt

# Copy application
COPY app.py .
COPY scripts/ ./scripts/
COPY runs/ ./runs/
COPY data/processed/ ./data/processed/
COPY .streamlit/ ./.streamlit/

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run:

```bash
# Build image
docker build -t precipitation-forecast .

# Run locally
docker run -p 8501:8501 precipitation-forecast

# Access at: http://localhost:8501
```

### Deploy to Docker Hub:

```bash
docker tag precipitation-forecast YOUR_USERNAME/precipitation-forecast
docker push YOUR_USERNAME/precipitation-forecast
```

---

## ☁️ Option 3: Cloud Platforms

### A. **Heroku** (Paid - $7/month minimum)

```bash
# Install Heroku CLI
# Create Procfile
echo "web: streamlit run app.py --server.port=\$PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### B. **Railway** (Free tier available)

1. Go to https://railway.app/
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Streamlit
5. Click "Deploy"

### C. **Google Cloud Run** (Pay-as-you-go)

```bash
# Build and deploy
gcloud run deploy precipitation-forecast \\
  --source . \\
  --platform managed \\
  --region us-central1 \\
  --allow-unauthenticated
```

### D. **AWS EC2** (Manual setup)

```bash
# SSH to EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip

# Clone repo
git clone https://github.com/YOUR_USERNAME/neuralhydrology-flood-forecast.git
cd neuralhydrology-flood-forecast

# Install requirements
pip3 install -r requirements_streamlit.txt

# Run with nohup (background)
nohup streamlit run app.py --server.port 8501 &

# Or use systemd for auto-restart (better)
```

---

## 🔧 Troubleshooting Deployment Issues

### Issue 1: Model file too large for GitHub
**Solution**: Use Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.nc"

# Add and commit
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### Issue 2: Out of memory on Streamlit Cloud
**Solution**: Optimize model loading

```python
@st.cache_resource
def load_model_and_data():
    # Already implemented - caches model in memory
    pass
```

### Issue 3: Slow deployment
**Solution**: Reduce PyTorch size in requirements

```txt
# Use CPU-only PyTorch
torch>=2.0.0+cpu
```

### Issue 4: NetCDF library errors
**Solution**: Already added to `packages.txt`:
- libhdf5-dev
- libnetcdf-dev

---

## 🔐 Environment Variables (for sensitive data)

If you need API keys or secrets:

### Streamlit Cloud:
1. Go to app settings
2. Add "Secrets" in TOML format:
```toml
[api_keys]
cds_api_key = "your-key-here"
```

### Docker:
```bash
docker run -e API_KEY=your-key -p 8501:8501 precipitation-forecast
```

---

## 📊 Performance Optimization

### 1. Enable Caching
Already implemented in app:
```python
@st.cache_resource  # Model loads once
@st.cache_data      # Data loads once
```

### 2. Compress Data
```bash
# Compress CSV to reduce size
gzip data/processed/era5/ganges_farakka_era5_daily.csv
```

### 3. Use Lighter Model
Train a quantized model (INT8):
```python
import torch.quantization
model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
```

---

## 🎯 Recommended Deployment: Streamlit Cloud

**Best for you because:**
- ✅ Free and fast
- ✅ No DevOps knowledge needed
- ✅ Auto-updates on git push
- ✅ Custom domain support
- ✅ Built-in analytics

**Next Steps:**
1. Create GitHub account (if you don't have one)
2. Push code to GitHub
3. Deploy to Streamlit Cloud (5 minutes)
4. Share your app link!

---

## 📱 Sharing Your App

Once deployed, you can:
- Share the URL: `https://your-app.streamlit.app`
- Embed in website: `<iframe src="https://your-app.streamlit.app"></iframe>`
- Add custom domain: Configure in Streamlit Cloud settings

---

## 🔄 Updating Your Deployed App

```bash
# Make changes to app.py
git add app.py
git commit -m "Update prediction logic"
git push

# Streamlit Cloud auto-deploys within 1-2 minutes!
```

---

## 💡 Tips for Production

1. **Add error handling** (already done in app)
2. **Monitor usage** with Streamlit analytics
3. **Set rate limits** if API-heavy
4. **Add Google Analytics** for tracking
5. **Enable HTTPS** (automatic on Streamlit Cloud)

---

## 🆘 Need Help?

- Streamlit Docs: https://docs.streamlit.io/
- Community Forum: https://discuss.streamlit.io/
- GitHub Issues: File issues in your repo

---

**Ready to deploy?** Start with Option 1 (Streamlit Cloud) - it's the easiest!
