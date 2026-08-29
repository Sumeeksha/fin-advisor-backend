"""News router — /api/news/{ticker}"""

from fastapi import APIRouter, HTTPException
from services.data_provider import get_news

router = APIRouter()


@router.get("/{ticker}")
def news(ticker: str):
    """Get latest news for a ticker."""
    items = get_news(ticker.upper())
    return {"ticker": ticker.upper(), "news": items}
