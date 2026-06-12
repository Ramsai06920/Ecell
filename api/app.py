from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI(
    title="SEC 10-K Financial Risk Classifier",
    description="Classifies SEC 10-K filings as high_risk, medium_risk, or low_risk",
    version="1.0.0"
)

# Load model and vectorizer once when server starts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model      = joblib.load(os.path.join(BASE_DIR, 'models', 'best_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'models', 'vectorizer.pkl'))
le         = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'))

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z ]', '', text)
    text = ' '.join(text.split())
    return text

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.get("/")
def root():
    return {"message": "SEC 10-K Risk Classifier API", 
            "docs": "/docs",
            "predict": "/predict"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Step 1: Clean the input text
    cleaned = clean_text(request.text)

    # Step 2: Vectorize using saved TF-IDF vectorizer
    features = vectorizer.transform([cleaned])

    # Step 3: Predict label
    pred_encoded = model.predict(features)[0]

    # Step 4: Get confidence score
    proba = model.predict_proba(features)[0]
    confidence = float(proba.max())

    # Step 5: Convert number back to label string
    label = le.inverse_transform([pred_encoded])[0]

    return PredictResponse(label=label, confidence=round(confidence, 2))

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost"}

