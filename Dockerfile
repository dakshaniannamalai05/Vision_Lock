FROM python:3.11-slim

# Install system dependencies required by OpenCV and Dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (8000 for local / cloud)
EXPOSE 8000

# Start FastAPI server via Uvicorn
CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]
