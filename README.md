# Customer Support Intent Classifier

This project uses the supplied `aiml_training_data.csv` to classify customer-support messages into intents.

## Important data note

The internship brief describes a course recommendation system, but the supplied CSV contains text-classification fields: `text`, `intent`, and `confidence`. This implementation matches the data that was actually provided. It is not a collaborative-filtering recommender.

## Setup

Copy `aiml_training_data.csv` into `data/`, or leave the original file at `C:/Users/supre/Downloads/aiml_training_data.csv`.

```powershell
python -m pip install -r requirements.txt
python train_model.py
uvicorn app:app --reload
```

The API runs at `http://127.0.0.1:8000`.

## Endpoints


## Model

The training script uses a stratified train/test split, word and bigram TF-IDF features, and Logistic Regression. It prints accuracy, a classification report, and a confusion matrix, then saves the pipeline to `artifacts/intent_classifier.joblib`.

## Notebook

Open `intent_classifier.ipynb` in VS Code or Jupyter and run the cells in order.

## ScholarS course recommendation prototype

`recommender_demo.py` and `recommender_api.py` provide a ScholarS course-recommendation prototype using substitute ratings with Surprise. Run `python recommender_demo.py` to train SVD and compare measured Hit Rate@5 and NDCG@5 against a popularity baseline.

Start the demo API with:

```powershell
uvicorn recommender_api:app --reload --port 8001
```

Then open `http://127.0.0.1:8001/docs`. The current metrics use substitute ratings and must be replaced with ScholarS learner-course data before being reported as client results.
