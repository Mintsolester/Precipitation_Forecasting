FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libnetcdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements_streamlit.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_streamlit.txt

# Copy application files
COPY app.py .
COPY scripts/ ./scripts/
COPY runs/ ./runs/
COPY data/processed/ ./data/processed/
COPY .streamlit/ ./.streamlit/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
