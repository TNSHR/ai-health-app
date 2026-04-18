from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (IMPORTANT for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class InputData(BaseModel):
    data: list

# Model definition
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 🔥 FIX: Safe paths (works locally + Render)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "cancer_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# Load model
model = Model()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Load scaler
scaler = joblib.load(SCALER_PATH)

@app.get("/")
def home():
    return {"message": "AI Health API Running"}

@app.post("/predict")
def predict(input: InputData):

    # Convert input to numpy
    data = np.array([input.data])

    # Scale input
    data = scaler.transform(data)

    # Convert to tensor
    x = torch.tensor(data, dtype=torch.float32)

    with torch.no_grad():
        pred = model(x).item()

    return {
        "prediction": round(pred, 4),
        "risk": "High" if pred > 0.5 else "Low"
    }