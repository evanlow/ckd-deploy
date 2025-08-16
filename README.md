# CKD Predictor API

This repository provides a FastAPI-based web service for predicting Chronic Kidney Disease (CKD) using a machine learning model. The service is designed for easy deployment and integration into data science workflows or clinical decision support systems.

## Features

- **REST API** for CKD prediction using a pre-trained machine learning model
- **/predict** endpoint accepts tabular data and returns predictions and probabilities
- **/health** endpoint for service health checks
- **/schema** endpoint to inspect the expected feature order
- **Automatic input normalization** (handles column name formatting and missing features)
- **Configurable model path and prediction threshold** via environment variables
- **Sample model artifacts** and configuration files included in the `artifacts/` directory

## How It Works

1. **Model Loading**: On startup, the API loads a pre-trained model (default: `artifacts/model.joblib`). The expected feature order is inferred from the model or from metadata/config files.
2. **Prediction**: The `/predict` endpoint accepts a JSON payload with a list of feature dictionaries. Input columns are normalized, missing features are filled with default values, and the model returns predictions and probabilities.
3. **Health and Schema**: The `/health` endpoint returns service status and expected feature count. The `/schema` endpoint returns the expected feature order for debugging and integration.

## API Endpoints

- `GET /health` — Returns service status and expected feature count
- `GET /schema` — Returns the expected feature order
- `POST /predict` — Accepts a JSON payload with rows of features, returns predictions and probabilities

## Example Usage

```json
POST /predict
{
  "rows": [
    {"age": 50, "blood_pressure": 80, ...},
    {"age": 65, "blood_pressure": 90, ...}
  ]
}
```

Response:
```json
{
  "pred": [0, 1],
  "proba": [0.12, 0.87],
  "threshold": 0.5,
  "received_columns": [...],
  "served_order": [...]
}
```

## Configuration

- `MODEL_PATH`: Path to the model file (default: `artifacts/model.joblib`)
- `DEFAULT_THRESHOLD`: Default probability threshold for binary classification (default: `0.50`)

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Running Locally

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Start the API:
   ```sh
   uvicorn app:app --reload
   ```

## Artifacts

The `artifacts/` directory contains sample model files, configuration, and metadata used for inference and schema validation.

## License

This project is provided for educational and research purposes. See LICENSE for details if present.
