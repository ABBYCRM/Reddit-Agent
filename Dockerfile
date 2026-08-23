FROM python:3.11-slim

WORKDIR /app

# Copy and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p /app/chroma_db

EXPOSE 8000

CMD ["python", "run.py"]
