FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/uploads /mlflow

EXPOSE 8080

# Use shell to expand PORT variable
CMD sh -c "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"