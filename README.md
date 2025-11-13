# Titanic Survival Prediction MLOps Pipeline

End-to-end MLOps workflow that ingests Titanic passenger records, engineers features, trains a Random Forest classifier, and serves real-time survival predictions with production-grade monitoring and data-drift detection.

## Architecture

- **Airflow ingestion**: `extract_data_from_gcp` DAG moves raw CSV data from Google Cloud Storage into a Postgres table.  
- **Data processing**: `DataProcessing` pipeline cleans data, engineers features, balances the target with SMOTE, and pushes features into Redis as an online feature store.  
- **Model training**: `TrainingModel` pulls features from Redis, performs hyperparameter tuning, trains a RandomForest, and stores the artifact under `artifacts/model/random_forest_model.pkl`.  
- **Inference service**: Flask app (`application.py`) loads the trained model, streams features from web forms, detects drift with Alibi Detect, and exposes Prometheus metrics.  
- **Observability**: Prometheus scrapes `/metrics`; Grafana dashboards can be provisioned via `docker-compose.yml`.  

```text
.
├── application.py             # Flask inference service with drift detection and metrics
├── pipeline/
│   └── training_pipeline.py    # Orchestrates ingestion → processing → training
├── src/
│   ├── data_ingestion.py       # Pull Titanic data from Postgres
│   ├── data_processing.py      # Feature engineering, SMOTE balancing, Redis writes
│   ├── model_training.py       # RandomForest training + hyperparameter search
│   ├── feature_store.py        # Redis client helpers
│   ├── logger.py               # Structured logging utility
│   └── custom_exception.py     # Unified exception formatting
├── dags/
│   ├── extract_data_from_gcp.py   # Airflow DAG to load raw data into Postgres
│   └── exampledag.py              # Astro starter example (kept for reference)
├── templates/ & static/        # Flask UI
├── docker-compose.yml          # Prometheus + Grafana stack
├── prometheus.yml              # Prometheus scrape config for the Flask app
├── config/                     # Paths and Postgres credentials
├── artifacts/                  # Raw data snapshots and trained model
└── tests/                      # Pytest DAG smoke tests
```

## Getting Started

### Prerequisites
- Python 3.10+
- Local or containerized instances of **Postgres** and **Redis**
- Optional: Docker (for Prometheus/Grafana), Apache Airflow (if you want to run the ingestion DAG)

### Installation
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Update `config/database_config.py` to match your Postgres instance. Ensure Redis is reachable at `localhost:6379` or adjust the defaults in `src/feature_store.py`.

## Data & Feature Pipelines

1. **Load raw data into Postgres**
   - Configure the Airflow connection `postgres_default` for your database.
   - Trigger the `extract_titanic_data` DAG (`dags/extract_data_from_gcp.py`) to pull `TitanicDataset.csv` from GCS into the `titanic` table.

   *Alternative*: place CSVs under `artifacts/raw/` manually if you prefer to skip Airflow.

2. **Run the training pipeline**
   ```bash
   python pipeline/training_pipeline.py
   ```
   This executes:
   - Database ingestion (`src/data_ingestion.py`)
   - Feature engineering + SMOTE balancing + Redis writes (`src/data_processing.py`)
   - RandomForest training and artifact persistence (`src/model_training.py`)

3. **Validate artifacts**
   - The trained model will be stored in `artifacts/model/random_forest_model.pkl`.
   - Redis should now contain passenger features keyed by `entity:{PassengerId}:features`.

## Serving Predictions

```bash
python application.py
```

- UI served at `http://localhost:5000/` (renders `templates/index.html`)
- Predictions posted to `/predict`
- Prometheus-formatted metrics at `/metrics`
- Drift detection (Kolmogorov–Smirnov) is triggered on every request, incrementing a Prometheus counter when drift is detected.

To expose metrics dashboards:
```bash
docker-compose up -d
```
Prometheus listens on `http://localhost:9090`, Grafana on `http://localhost:3000` (default credentials `admin/admin`).

## Testing

Smoke tests for DAG imports and configuration:
```bash
pytest tests
```

## Project Highlights
- **Feature Store Integration**: Redis keeps online features synchronized with training data for low-latency lookups.
- **Data Drift Monitoring**: Alibi Detect’s KS drift detector runs in real time against a reference window populated from Redis.
- **Operational Metrics**: Prometheus counters track inference volume and drift events; ready for Grafana dashboards.
- **Automation Ready**: Airflow DAG and Python pipeline scripts cover ingestion-to-training automation for reproducibility.

## Roadmap Ideas
- Add CI/CD for automatic testing and deployment (GitHub Actions).
- Parameterize configuration via environment variables instead of hardcoded values.
- Extend monitoring with model accuracy tracking and alerting via Grafana.
