# AI Health Risk Predictor

## Description
Predicts cancer risk using machine learning.

## Features
- PyTorch model
- FastAPI backend
- Real-time prediction

## Run Locally
uvicorn api.main:app --reload

## Endpoint
POST /predict

## Input
{
  "data": [30 values]
}

## Output
{
  "prediction": 0.73,
  "risk": "High"
}