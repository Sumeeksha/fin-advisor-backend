"""Advice router — /api/advice/{ticker}"""

from fastapi import APIRouter, HTTPException
from services.data_provider import get_quote, get_history
from services.indicators import get_all_indicators
from services.advisor import generate_advice

router = APIRouter()


@router.get("/{ticker}")
def advice(ticker: str):
    """Generate AI-powered Buy/Hold/Sell advice for a ticker."""
    quote = get_quote(ticker.upper())
    if "error" in quote:
        raise HTTPException(status_code=404, detail=quote["error"])

    df = get_history(ticker.upper(), "3M")
    indicators = get_all_indicators(df) if df is not None and not df.empty else {"error": "No data"}

    result = generate_advice(quote, indicators)
    result["ticker"] = ticker.upper()
    result["quote"] = quote
    return result
