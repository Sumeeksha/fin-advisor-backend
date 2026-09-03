"""Indicators router — /api/indicators/{ticker}"""

from fastapi import APIRouter, HTTPException, Query
from services.data_provider import get_history
from services.indicators import get_all_indicators

router = APIRouter()


@router.get("/{ticker}")
def indicators(ticker: str, period: str = Query("3M", pattern="^(1M|3M|1Y|5Y)$")):
    """Compute and return all technical indicators for a ticker."""
    df = get_history(ticker.upper(), period)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    result = get_all_indicators(df)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    result["ticker"] = ticker.upper()
    result["period"] = period
    return result
