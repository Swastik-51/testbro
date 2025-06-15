from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and vectorizer
vectorizer = joblib.load("models/ahirr_vectorizer.pkl")
model = joblib.load("models/ahirr_model.pkl")

# Input schema
class JobListing(BaseModel):
    title: str = ""
    company_profile: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""

@app.post("/predict")
def predict_job(job: JobListing):
    # Combine all text fields
    text = " ".join([
        job.title, job.company_profile,
        job.description, job.requirements,
        job.benefits
    ]).lower().strip()

    # Vectorize and predict
    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0, 1]
    label = "Fraud" if prob > 0.5 else "Genuine"

    return {
        "prediction": label,
        "fraud_probability": round(float(prob) * 100, 2)
    }
