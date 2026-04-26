FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/uploads /mlflow

EXPOSE 8000

# Railway sets PORT dynamically - use it!
CMD sh -c "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"