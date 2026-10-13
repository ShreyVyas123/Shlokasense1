from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import ShlokaSensePipeline


app = FastAPI(
    title="ShlokaSense API",
    description="AI-powered Bhagavad Gita guidance system",
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    text: str


# ---------------------------------------------------------
# LOAD ML PIPELINE ONCE
# ---------------------------------------------------------

print("=" * 60)
print("SHLOKASENSE - STARTING BACKEND")
print("=" * 60)

pipeline = ShlokaSensePipeline()

print("Backend ready!")


# ---------------------------------------------------------
# ROOT
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "ShlokaSense API is running",
        "status": "online",
    }


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "distilbert-emotion",
        "retrieval": "faiss",
    }


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    try:

        result = pipeline.predict(
            text,
            top_k=3,
        )

        return result

    except Exception as error:

        print("Prediction error:", error)

        raise HTTPException(
            status_code=500,
            detail="Failed to process the request.",
        )