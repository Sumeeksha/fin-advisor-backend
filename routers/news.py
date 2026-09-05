"""News router — /api/news/{ticker}"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from services.news_service import fetch_all_news

router = APIRouter()


@router.get("/{ticker}")
def news(
    ticker: str,
    types: Optional[str] = Query(
        default="news,sec,rss",
        description=(
            "Comma-separated source types to include. "
            "Options: news (Finnhub/yfinance), sec (SEC EDGAR filings), rss (Google/Yahoo RSS). "
            "Example: ?types=news,sec"
        ),
    ),
    limit: int = Query(default=20, ge=1, le=50, description="Maximum items to return (1–50)."),
):
    """
    Get unified news feed for a ticker, including live market news,
    SEC EDGAR regulatory filings (10-K, 10-Q, 8-K), and RSS feeds.

    Every returned item contains a canonical URL pointing directly to the
    original article or official SEC filing.
    """
    ticker_clean = ticker.upper().strip()
    if not ticker_clean:
        raise HTTPException(status_code=400, detail="Ticker symbol is required.")

    type_list = [t.strip().lower() for t in types.split(",") if t.strip()]
    valid_types = {"news", "sec", "rss"}
    invalid = set(type_list) - valid_types
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid type(s): {invalid}. Valid options: {valid_types}",
        )

    items = fetch_all_news(ticker=ticker_clean, types=type_list, limit=limit)
    return {
        "ticker": ticker_clean,
        "types": type_list,
        "count": len(items),
        "news": items,
    }
