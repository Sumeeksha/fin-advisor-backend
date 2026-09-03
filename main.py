"""
Financial Advisory App — FastAPI Backend
Entry point: main.py
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routers import stocks, indicators, advice, forecast, news, insight, auth

load_dotenv()

app = FastAPI(
    title="Financial Advisory API",
    description="Real-time stock analysis, technical indicators, and AI-powered investment advice",
    version="1.0.0",
)

# CORS — allow frontend dev server
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://fin-advisor-ui-87993570672.us-central1.run.app",  # Added your Cloud Run UI domain
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(stocks.router, prefix="/api/stocks", tags=["Stocks"])
app.include_router(indicators.router, prefix="/api/indicators", tags=["Indicators"])
app.include_router(advice.router, prefix="/api/advice", tags=["Advice"])
app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecast"])
app.include_router(news.router, prefix="/api/news", tags=["News"])
app.include_router(insight.router, prefix="/api/insight", tags=["Insight"])



@app.get("/")
def root():
    return {"status": "ok", "message": "Financial Advisory API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}
