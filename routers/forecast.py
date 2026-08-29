"""Forecast router — /api/forecast/{ticker}"""

from fastapi import APIRouter, HTTPException, Query
from services.data_provider import get_history
from services.forecaster import generate_forecast

router = APIRouter()


@router.get("/{ticker}")
def forecast(ticker: str, days: int = Query(14, ge=7, le=30)):
    """Generate price forecast for a ticker."""
    df = get_history(ticker.upper(), "1Y")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    result = generate_forecast(df, days=days)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    result["ticker"] = ticker.upper()
    return result
