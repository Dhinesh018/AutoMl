# 🤖 LLM-Augmented AutoML System

> An intelligent AutoML pipeline that uses LLM reasoning to select the best ML models — achieving 70% faster training than traditional AutoML.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-orange.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

---

## 🎯 What Makes This Different

| Traditional AutoML | This System |
|-------------------|-------------|
| Trains ALL models blindly | LLM analyzes & selects best 1-3 models |
| Wastes compute resources | ~70% faster training time |
| No explainability | LLM reasoning logged to MLflow |

---

## ✨ Key Features

- **🧠 LLM Model Selection** - Groq API (Llama 3.3 70B) profiles your dataset and intelligently selects promising algorithms
- **📊 Full MLOps Pipeline** - MLflow experiment tracking, model registry, versioning, promotion, rollback
- **⚡ Background Training** - Non-blocking async training with real-time progress monitoring
- **🚀 Production Ready** - Auto-promotion, one-click rollback, version comparison
- **📈 5 ML Algorithms** - RandomForest, XGBoost, LightGBM, ElasticNet, LinearRegression

---

## 🏗️ Architecture
```
Dataset Upload → LLM Profiles Data → Selects Best Models
    → Trains Subset → Evaluates → Registers Best
    → Auto-Promotes to Production → Serves Predictions
```

**Tech Stack:**
- **Backend:** FastAPI, Python 3.11
- **ML:** scikit-learn, XGBoost, LightGBM
- **LLM:** Groq API (Llama 3.3 70B)
- **MLOps:** MLflow (tracking + registry)
- **Database:** SQLite
- **Deployment:** Docker, Docker Compose

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/automl-assistant.git
cd automl-assistant
```

2. **Set up environment variables**
```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
echo "MLFLOW_TRACKING_URI=http://mlflow:5000" >> .env
```

3. **Start the system**
```bash
docker-compose up -d
```

4. **Verify it's running**
```bash
docker ps
# Both containers should be running: automl-api, mlflow-server
```

5. **Access the interfaces**
- **API Documentation:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5001

---

## 📖 Usage

### 1. Upload a Dataset
```bash
curl -X POST "http://localhost:8000/datasets/upload" \
  -F "file=@your_data.csv" \
  -F "target_column=price"
```

**Response:**
```json
{
  "dataset_id": "dataset_20260316_123456_abc123",
  "filename": "your_data.csv",
  "num_rows": 1000,
  "num_columns": 15,
  "target_column": "price"
}
```

### 2. Train a Model
```bash
curl -X POST "http://localhost:8000/train?dataset_id=dataset_20260316_123456_abc123&target_column=price"
```

**Response:**
```json
{
  "job_id": "job_xyz789",
  "status": "queued",
  "message": "Training started. Poll /train/status/job_xyz789 for progress."
}
```

### 3. Monitor Training Progress
```bash
curl "http://localhost:8000/train/status/job_xyz789"
```

**Response (while running):**
```json
{
  "job_id": "job_xyz789",
  "status": "running",
  "progress": 60,
  "current_step": "Training XGBoost (model 2/3)"
}
```

**Response (completed):**
```json
{
  "job_id": "job_xyz789",
  "status": "completed",
  "progress": 100,
  "result": {
    "best_model": "XGBoost",
    "best_score": 0.924,
    "model_version": 5
  }
}
```

### 4. Restart API to Load New Model
```bash
docker-compose restart api
```

### 5. Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature1": 10,
      "feature2": 20,
      "feature3": 30
    }
  }'
```

**Response:**
```json
{
  "prediction": 215000.0,
  "model_name": "llm_automl_tabular_model",
  "model_version": 5,
  "model_stage": "Production"
}
```

---

## 📁 Project Structure
```
automl-assistant/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   ├── upload.py            # Dataset upload logic
│   │   └── schemas.py           # Pydantic models
│   ├── automl/
│   │   ├── automl_engine.py     # Model training loop
│   │   ├── train.py             # Training orchestration
│   │   ├── data_loader.py       # Dataset loading
│   │   ├── data_profiler.py     # Dataset profiling
│   │   ├── preprocessor.py      # Feature engineering
│   │   └── evaluate.py          # Model evaluation
│   ├── llm/
│   │   ├── real_llm.py          # Groq API integration
│   │   └── llm_prompts.py       # LLM system prompts
│   ├── jobs/
│   │   ├── job_store.py         # Training job tracking
│   │   └── training_jobs.py     # Background training
│   ├── utils/
│   │   ├── logger.py            # Structured logging
│   │   └── exceptions.py        # Custom exceptions
│   └── config.py                # Configuration
├── data/
│   └── medium_house_prices.csv  # Sample dataset
├── Dockerfile                   # API container
├── docker-compose.yml           # Multi-container setup
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
└── README.md                    # This file
```

---

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/datasets/upload` | POST | Upload training dataset |
| `/train` | POST | Start training job |
| `/train/status/{job_id}` | GET | Check training progress |
| `/train/jobs` | GET | List all training jobs |
| `/predict` | POST | Get prediction |
| `/models/versions` | GET | List model versions |
| `/models/promote/{version}` | POST | Promote model to stage |
| `/models/rollback` | POST | Rollback to previous model |
| `/models/compare` | GET | Compare two versions |
| `/health` | GET | System health check |

**Full API documentation:** http://localhost:8000/docs

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM | Required |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://mlflow:5000` |

### Training Configuration

Edit `configs/train_config.json` to customize:
- Train/test split ratio
- Random seed
- Model hyperparameters

---

## 🐳 Docker Commands
```bash
# Start services
docker-compose up -d

# View logs
docker logs automl-api
docker logs mlflow-server

# Restart API (after training)
docker-compose restart api

# Stop services
docker-compose down

# Rebuild containers
docker-compose up -d --build
```

---

## 🧪 Development

### Local Setup (without Docker)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY=your_key
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Start MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Start API
uvicorn src.api.main:app --reload
```

---

## 📊 How the LLM Selection Works

1. **Dataset Profiling** - System analyzes:
   - Number of features
   - Data types (numeric, categorical)
   - Target distribution
   - Missing values
   - Feature correlations

2. **LLM Analysis** - Groq API receives profile and decides:
   - Which models suit this dataset type
   - Which models to skip (saves time)
   - Reasoning logged to MLflow

3. **Training** - Only selected models are trained

4. **Result** - ~70% faster than training all 5 models

**Example LLM Decision:**
```json
{
  "selected_models": ["XGBoost", "LightGBM"],
  "skipped_models": ["LinearRegression", "ElasticNet", "RandomForest"],
  "reasoning": "Dataset has 50+ features with non-linear patterns. Tree-based models (XGBoost, LightGBM) will likely outperform linear models. RandomForest skipped as gradient boosting typically performs better on tabular data."
}
```

---

## 🚧 Roadmap

- [x] Core AutoML engine
- [x] LLM model selection
- [x] MLflow integration
- [x] Docker deployment
- [x] API documentation
- [ ] React frontend (in progress)
- [ ] Cloud deployment
- [ ] User authentication
- [ ] Dataset versioning

---

## 🤝 Contributing

This is a final year project, but suggestions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **Groq** - For free LLM API access
- **MLflow** - For excellent ML lifecycle management
- **FastAPI** - For amazing API framework

---

## 📧 Contact

**Your Name** - [dineshdk26072004@gmail.com](dineshdk26072004@gmail.com)

**Project Link:** [https://github.com/Dhinesh018/automl-assistant](https://github.com/Dhinesh018/automl-assistant)

---

**⭐ Star this repo if you found it helpful!**
```