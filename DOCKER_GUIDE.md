# Docker Operations Guide

## Starting the Stack
```bash
docker-compose up           # Start with logs
docker-compose up -d        # Start in background
docker-compose up --build   # Rebuild and start
```

## Stopping
```bash
docker-compose down         # Stop (keeps data)
docker-compose down -v      # Stop + delete data ⚠️
```

## Training
```bash
docker exec automl-api python run_train_test.py
```

## Checking Status
```bash
docker ps                   # Running containers
docker logs automl-api      # API logs
docker logs mlflow-server   # MLflow logs
```

## Access Points

- API: http://localhost:8000
- MLflow: http://localhost:5001
- API Docs: http://localhost:8000/docs

## Troubleshooting
```bash
docker-compose restart api      # Restart API
docker-compose restart mlflow   # Restart MLflow
```

## Data Persistence

- Stored in: `mlflow_data/`
- Survives restarts ✅
- Deleted with `docker-compose down -v` ⚠️