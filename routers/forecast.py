"""Forecast router — /api/forecast/{ticker}"""

from fastapi import APIRouter, HTTPException, Query
from services.data_provider import get_history, get_quote
from services.forecaster import generate_forecast

router = APIRouter()


@router.get("/{ticker}")
def forecast(ticker: str, days: int = Query(14, ge=7, le=30)):
    """Generate price forecast for a ticker."""
    ticker_upper = ticker.upper()
    df = get_history(ticker_upper, "1Y")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    # Fetch latest live price to inject into the ARIMA series
    quote = get_quote(ticker_upper)
    latest_price = quote.get("price") if quote else None

    result = generate_forecast(df, days=days, latest_price=latest_price)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    result["ticker"] = ticker_upper
    return result
