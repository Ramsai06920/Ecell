# Ecell
Ecell induction task
# SEC 10-K Financial Risk Classifier

## Project Overview
End-to-end ML pipeline that classifies SEC 10-K filings 
as high_risk, medium_risk, or low_risk using NLP and boosting models.

## Dataset
winterForestStump/10-K_sec_filings (HuggingFace)

## Classification Target
Financial risk classification derived from domain-specific 
lexicon of risk and growth indicators, normalized by document length.

## Pipeline
- Stage 1: preprocess.py — data loading, text cleaning, label generation
- Stage 2: features.py   — TF-IDF vectorization, train/test split
- Stage 3: train.py      — XGBoost, AdaBoost, CatBoost training
- Stage 4: evaluate.py   — metrics, confusion matrix, model comparison
- Stage 5: api/app.py    — FastAPI prediction endpoint

## Results
| Model    | Accuracy | F1 Score |
|----------|----------|----------|
| XGBoost  | 89.54%   | 0.8944   |
| AdaBoost | 86.22%   | 0.8605   |
| CatBoost | 70.92%   | 0.7065   |

Best model: XGBoost

## Run the API
uvicorn api.app:app --reload

## Endpoint
POST /predict
Input:  {"text": "..."}
Output: {"label": "high_risk", "confidence": 0.91}
