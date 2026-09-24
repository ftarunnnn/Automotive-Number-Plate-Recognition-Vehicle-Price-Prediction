FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn phase_9_backend_integration.app:app --host 0.0.0.0 --port 8000 & streamlit run phase_10_frontend_deployment/app_ui.py --server.port 8501 --server.address 0.0.0.0"]
