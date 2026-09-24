from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from recommender_demo import ARTIFACT_PATH, TOP_K, evaluate, recommend_svd

app = FastAPI(title="ScholarS Course Recommendation Prototype", version="1.0.0")


def load_artifact():
    if not ARTIFACT_PATH.exists():
        evaluate()
    return joblib.load(ARTIFACT_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "dataset": "ScholarS course recommendation prototype"}


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = TOP_K) -> dict[str, object]:
    if not 1 <= k <= 20:
        raise HTTPException(status_code=400, detail="k must be between 1 and 20")
    artifact = load_artifact()
    trainset = artifact["trainset"]
    if user_id not in trainset._raw2inner_id_users:
        raise HTTPException(status_code=404, detail="Unknown learner user_id")
    items = recommend_svd(artifact["model"], trainset, user_id, k)
    return {
        "user_id": user_id,
        "recommendations": items,
        "dataset": "ScholarS course recommendation prototype",
        "data_note": "Replace substitute ratings with ScholarS learner-course history.",
    }
