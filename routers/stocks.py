"""Stocks router — /api/stocks/{ticker}"""

from fastapi import APIRouter, HTTPException, Query
from typing import List
from services.data_provider import get_quote, search_tickers, get_company_info, get_history, get_market_ribbon

router = APIRouter()


@router.get("/search")
def search(q: str = Query(..., min_length=1)):
    """Search tickers by name or symbol."""
    results = search_tickers(q)
    return {"results": results}


@router.get("/market-ribbon")
def market_ribbon():
    """Get live market index and trending quotes for header marquee."""
    return get_market_ribbon()


@router.get("/{ticker}/quote")
def quote(ticker: str):
    """Get live quote for a ticker."""
    data = get_quote(ticker.upper())
    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
    return data


@router.get("/{ticker}/info")
def company_info(ticker: str):
    """Get company fundamentals."""
    data = get_company_info(ticker.upper())
    return data


@router.get("/{ticker}/history")
def history(ticker: str, period: str = Query("1M", pattern="^(1D|1W|1M|3M|1Y|5Y)$")):
    """Get OHLCV price history."""
    df = get_history(ticker.upper(), period)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No history data for {ticker}")

    records = []
    for ts, row in df.iterrows():
        date_str = str(ts.date()) if hasattr(ts, "date") else str(ts)
        records.append({
            "date": date_str,
            "timestamp": int(ts.timestamp()) if hasattr(ts, "timestamp") else 0,
            "open": round(float(row.get("Open", 0)), 2),
            "high": round(float(row.get("High", 0)), 2),
            "low": round(float(row.get("Low", 0)), 2),
            "close": round(float(row.get("Close", 0)), 2),
            "volume": int(row.get("Volume", 0)) if row.get("Volume") else 0,
        })

    return {"ticker": ticker.upper(), "period": period, "data": records}
