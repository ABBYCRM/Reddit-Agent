FROM python:3.11-slim

WORKDIR /app

# Copy and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories and run as an unprivileged user
RUN mkdir -p /app/data/chroma_db && useradd --create-home appuser && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
