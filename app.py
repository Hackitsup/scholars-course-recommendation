from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train_model import MODEL_PATH, train_and_save

app = FastAPI(title="Customer Support Intent API", version="1.0.0")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    text: str
    intent: str
    confidence: float


def load_model():
    if not MODEL_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model = load_model()
        probabilities = model.predict_proba([request.text])[0]
        best_index = probabilities.argmax()
        intent = model.classes_[best_index]
        return PredictionResponse(
            text=request.text,
            intent=intent,
            confidence=float(probabilities[best_index]),
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
