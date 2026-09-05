"""
Data Provider Service — Unified stock data fetching with fallback chain.
Priority: yfinance (no key) → Finnhub (live quotes) → Alpha Vantage (technical)
Automatic sandbox/mock mode fallback when all live sources rate-limit or fail.
"""

import os
import time
import logging
import random
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)

# ── In-Memory Cache ───────────────────────────
_quote_cache = {}
_history_cache = {}
_search_cache = {}

# ── Rate limit / block tracking ───────────────
# Track if Yahoo Finance is rate-limiting us (to fail fast instead of hanging)
_yfinance_blocked_until = 0.0

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FMP_KEY = os.getenv("FMP_API_KEY", "")

PERIOD_MAP = {
    "1D": ("1d", "5m"),
    "1W": ("5d", "30m"),
    "1M": ("1mo", "1h"),
    "3M": ("3mo", "1d"),
    "1Y": ("1y", "1d"),
    "5Y": ("5y", "1wk"),
}


def _check_yfinance_blocked() -> bool:
    """Check if yfinance is temporarily marked as blocked."""
    global _yfinance_blocked_until
    return time.time() < _yfinance_blocked_until


def _mark_yfinance_blocked():
    """Mark yfinance as blocked for 15 minutes to fail-fast."""
    global _yfinance_blocked_until
    logger.warning("yfinance rate-limited/blocked. Entering Sandbox Mode for 15 minutes.")
    _yfinance_blocked_until = time.time() + 900.0


def get_quote(ticker: str) -> Dict[str, Any]:
    """Get live quote data with fallback chain and mock support."""
    ticker = ticker.upper().strip()
    cache_key = f"quote_{ticker}"
    
    # Try TTL cache
    if cache_key in _quote_cache:
        cached_val, ts = _quote_cache[cache_key]
        if time.time() - ts < 30:
            return cached_val

    result = None

    # Try yfinance first (unless marked blocked)
    if not _check_yfinance_blocked():
        result = _quote_yfinance(ticker)

    # Try Finnhub if yfinance fails
    if (not result or "error" in result) and FINNHUB_KEY and FINNHUB_KEY != "YOUR_FINNHUB_KEY_HERE":
        result = _quote_finnhub(ticker)

    # Fallback to sandbox mode
    if not result or "error" in result:
        result = _mock_quote(ticker)

    _quote_cache[cache_key] = (result, time.time())
    return result


def get_history(ticker: str, period: str = "1M") -> pd.DataFrame:
    """Get OHLCV history with yfinance and mock fallback."""
    ticker = ticker.upper().strip()
    cache_key = f"history_{ticker}_{period}"
    
    if cache_key in _history_cache:
        cached_val, ts = _history_cache[cache_key]
        if time.time() - ts < 300:
            return cached_val

    df = None
    
    # Try yfinance unless marked blocked
    if not _check_yfinance_blocked():
        yf_period, yf_interval = PERIOD_MAP.get(period, ("1mo", "1d"))
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period=yf_period, interval=yf_interval, auto_adjust=True)
            if df is not None and not df.empty and len(df) > 5:
                df.index = pd.to_datetime(df.index)
                _history_cache[cache_key] = (df, time.time())
                return df
        except Exception as e:
            logger.warning(f"yfinance history failed for {ticker}: {e}")
            _mark_yfinance_blocked()

    # Fallback to mock history
    df = _mock_history(ticker, period)
    _history_cache[cache_key] = (df, time.time())
    return df


def search_tickers(query: str) -> List[Dict[str, str]]:
    """Search for ticker symbols matching a query string."""
    cache_key = f"search_{query.lower()}"
    if cache_key in _search_cache:
        return _search_cache[cache_key]

    results = []
    
    if not _check_yfinance_blocked():
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&newsCount=0&quotesCount=10"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = requests.get(url, headers=headers, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("quotes", [])[:10]:
                    if item.get("quoteType") in ("EQUITY", "ETF", "MUTUALFUND"):
                        results.append({
                            "symbol": item.get("symbol", ""),
                            "name": item.get("longname") or item.get("shortname", ""),
                            "type": item.get("quoteType", ""),
                            "exchange": item.get("exchDisp", ""),
                        })
            elif resp.status_code == 429:
                _mark_yfinance_blocked()
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            _mark_yfinance_blocked()

    if not results and FINNHUB_KEY and FINNHUB_KEY != "YOUR_FINNHUB_KEY_HERE":
        try:
            url = f"https://finnhub.io/api/v1/search?q={query}&token={FINNHUB_KEY}"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("result", [])[:10]:
                    results.append({
                        "symbol": item.get("symbol", ""),
                        "name": item.get("description", ""),
                        "type": item.get("type", ""),
                        "exchange": "",
                    })
        except Exception as e:
            logger.warning(f"Finnhub search failed: {e}")

    if not results:
        popular_list = [
            {"symbol": "AAPL", "name": "Apple Inc.", "type": "EQUITY", "exchange": "NASDAQ"},
            {"symbol": "TSLA", "name": "Tesla, Inc.", "type": "EQUITY", "exchange": "NASDAQ"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "type": "EQUITY", "exchange": "NASDAQ"},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "type": "EQUITY", "exchange": "NASDAQ"},
            {"symbol": "AMZN", "name": "Amazon.com, Inc.", "type": "EQUITY", "exchange": "NASDAQ"},
            {"symbol": "NVDA", "name": "NVIDIA Corporation", "type": "EQUITY", "exchange": "NASDAQ"},
            {"symbol": "META", "name": "Meta Platforms, Inc.", "type": "EQUITY", "exchange": "NASDAQ"},
            {"symbol": "NFLX", "name": "Netflix, Inc.", "type": "EQUITY", "exchange": "NASDAQ"},
        ]
        q = query.upper()
        results = [
            item for item in popular_list
            if q in item["symbol"] or q in item["name"].upper()
        ]
        if not results and len(query) <= 5 and query.isalpha():
            results.append({
                "symbol": q,
                "name": f"{q} Corporation (Sandbox Mode)",
                "type": "EQUITY",
                "exchange": "NASDAQ",
            })

    _search_cache[cache_key] = results
    return results


def get_company_info(ticker: str) -> Dict[str, Any]:
    """Get company fundamentals via yfinance or mock."""
    ticker = ticker.upper().strip()
    
    if not _check_yfinance_blocked():
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            if not info or "longBusinessSummary" not in info:
                raise ValueError("No business summary, possibly blocked")
            return {
                "name": info.get("longName") or info.get("shortName", ticker),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "description": info.get("longBusinessSummary", ""),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "eps": info.get("trailingEps"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
                "website": info.get("website", ""),
                "country": info.get("country", ""),
            }
        except Exception as e:
            logger.warning(f"Company info failed for {ticker}, falling back: {e}")
            _mark_yfinance_blocked()

    # Mock company details
    random.seed(hash(ticker))
    pe = random.uniform(15, 35)
    eps = random.uniform(2, 10)
    base = _get_base_price(ticker)
    return {
        "name": f"{ticker} Corporation",
        "sector": "Technology" if ticker in ["AAPL", "NVDA", "MSFT"] else "Automotive" if ticker == "TSLA" else "Financial Services",
        "industry": "Consumer Electronics" if ticker == "AAPL" else "Software" if ticker == "MSFT" else "Internet Content & Information",
        "description": f"This is a sandbox metadata card for {ticker} Corporation. Real-time Yahoo Finance metadata is currently rate-limited, so FinAdvisor has generated this portfolio summary and financial ratios mathematically.",
        "market_cap": random.randint(100_000_000_000, 2_500_000_000_000),
        "pe_ratio": round(pe, 2),
        "eps": round(eps, 2),
        "dividend_yield": round(random.uniform(0.005, 0.025), 4),
        "beta": round(random.uniform(0.8, 1.6), 2),
        "52w_high": round(random.uniform(1.1, 1.25) * base, 2),
        "52w_low": round(random.uniform(0.75, 0.9) * base, 2),
        "avg_volume": random.randint(20_000_000, 100_000_000),
        "website": f"https://www.{ticker.lower()}.com",
        "country": "United States",
    }


def get_news(ticker: str) -> List[Dict[str, Any]]:
    """Get recent news for a ticker with news generation fallback."""
    ticker = ticker.upper().strip()
    news_items = []

    # Try Finnhub first (better news quality)
    if FINNHUB_KEY and FINNHUB_KEY != "YOUR_FINNHUB_KEY_HERE":
        try:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            url = (
                f"https://finnhub.io/api/v1/company-news"
                f"?symbol={ticker}&from={from_date}&to={to_date}&token={FINNHUB_KEY}"
            )
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                for item in resp.json()[:8]:
                    news_items.append({
                        "headline": item.get("headline", ""),
                        "summary": item.get("summary", ""),
                        "source": item.get("source", ""),
                        "url": item.get("url", ""),
                        "datetime": item.get("datetime", 0),
                        "image": item.get("image", ""),
                    })
        except Exception as e:
            logger.warning(f"Finnhub news failed: {e}")

    # Fallback: yfinance news (unless blocked)
    if not news_items and not _check_yfinance_blocked():
        try:
            tk = yf.Ticker(ticker)
            for item in (tk.news or [])[:8]:
                content = item.get("content", {})
                news_items.append({
                    "headline": content.get("title", item.get("title", "")),
                    "summary": content.get("summary", ""),
                    "source": content.get("provider", {}).get("displayName", ""),
                    "url": content.get("canonicalUrl", {}).get("url", ""),
                    "datetime": int(datetime.fromisoformat(
                        content.get("pubDate", datetime.now().isoformat())
                        .replace("Z", "+00:00")
                    ).timestamp()) if content.get("pubDate") else int(time.time()),
                    "image": "",
                })
        except Exception as e:
            logger.warning(f"yfinance news failed: {e}")
            _mark_yfinance_blocked()

    # Fallback to realistic mock news
    if not news_items:
        headlines = [
            f"{ticker} announces quarterly earnings beating analyst expectations",
            f"Market trends: Why investors are looking closely at {ticker} stock today",
            f"Regulatory changes and their potential impact on {ticker} and competitors",
            f"Tech analysts update price target for {ticker} following key product launch",
            f"Global supply chain improvements expected to boost margins for {ticker}",
            f"{ticker} CEO speaks at annual technology conference detailing future vision"
        ]
        random.seed(int(time.time() // 3600) + hash(ticker))
        selected = random.sample(headlines, min(4, len(headlines)))
        for i, hl in enumerate(selected):
            news_items.append({
                "headline": hl,
                "summary": f"This is a sandbox-generated news summary for {ticker} describing the potential impact of market dynamics, interest rates, and earnings cycles on stock price indicators.",
                "source": "FinAdvisor News",
                "url": "#",
                "datetime": int(time.time()) - (i * 3600 * 4),
                "image": ""
            })

    return news_items


# ── Private helpers ───────────────────────────

def _get_base_price(ticker: str) -> float:
    base_prices = {
        "AAPL": 185.0, "TSLA": 220.0, "MSFT": 415.0, "GOOGL": 170.0,
        "AMZN": 180.0, "NVDA": 120.0, "META": 470.0, "NFLX": 620.0
    }
    return base_prices.get(ticker, 100.0)


def _quote_yfinance(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        tk = yf.Ticker(ticker)
        fi = tk.fast_info
        price = fi.last_price
        prev_close = fi.previous_close or fi.open
        if price is None:
            return None

        open_price = fi.open
        day_high = fi.day_high
        day_low = fi.day_low
        volume = fi.last_volume
        market_cap = fi.market_cap

        change = (price - prev_close) if prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0

        return {
            "ticker": ticker.upper(),
            "name": ticker,
            "price": round(float(price), 2),
            "prev_close": round(float(prev_close), 2) if prev_close else None,
            "open": round(float(open_price), 2) if open_price else None,
            "day_high": round(float(day_high), 2) if day_high else None,
            "day_low": round(float(day_low), 2) if day_low else None,
            "volume": int(volume) if volume else None,
            "market_cap": int(market_cap) if market_cap else None,
            "change": round(float(change), 2),
            "change_pct": round(float(change_pct), 2),
            "currency": "USD",
            "exchange": "NASDAQ",
            "source": "yfinance",
        }
    except Exception as e:
        logger.warning(f"_quote_yfinance failed for {ticker}: {e}")
        _mark_yfinance_blocked()
        return None


def _quote_finnhub(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_KEY}"
        resp = requests.get(url, timeout=3)
        if resp.status_code != 200:
            return None
        d = resp.json()
        price = d.get("c")
        prev_close = d.get("pc")
        if not price:
            return None
        change = d.get("d", 0)
        change_pct = d.get("dp", 0)
        return {
            "ticker": ticker.upper(),
            "name": ticker,
            "price": round(float(price), 2),
            "prev_close": round(float(prev_close), 2) if prev_close else None,
            "open": round(float(d.get("o", 0)), 2),
            "day_high": round(float(d.get("h", 0)), 2),
            "day_low": round(float(d.get("l", 0)), 2),
            "volume": None,
            "market_cap": None,
            "change": round(float(change), 2),
            "change_pct": round(float(change_pct), 2),
            "currency": "USD",
            "exchange": "",
            "source": "finnhub",
        }
    except Exception as e:
        logger.warning(f"_quote_finnhub failed for {ticker}: {e}")
        return None


def _mock_quote(ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper()
    base = _get_base_price(ticker)
    
    random.seed(int(time.time() // 60) + hash(ticker))
    change_pct = random.uniform(-2.5, 2.8)
    price = base * (1 + change_pct / 100)
    prev_close = base
    change = price - prev_close
    
    return {
        "ticker": ticker,
        "name": f"{ticker} Corporation (Sandbox Mode)",
        "price": round(price, 2),
        "prev_close": round(prev_close, 2),
        "open": round(prev_close * random.uniform(0.99, 1.01), 2),
        "day_high": round(max(price, prev_close) * random.uniform(1.0, 1.02), 2),
        "day_low": round(min(price, prev_close) * random.uniform(0.98, 1.0), 2),
        "volume": random.randint(10_000_000, 80_000_000),
        "market_cap": random.randint(50_000_000_000, 3_000_000_000_000),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "currency": "USD",
        "exchange": "NASDAQ (MOCK)",
        "source": "Sandbox Mode (Rate Limited / No Key)",
    }


def _mock_history(ticker: str, period: str) -> pd.DataFrame:
    ticker = ticker.upper()
    
    days_map = {
        "1D": 100,
        "1W": 50,
        "1M": 30,
        "3M": 90,
        "1Y": 252,
        "5Y": 260,
    }
    num_points = days_map.get(period, 90)
    base = _get_base_price(ticker)
    
    end_date = datetime.now()
    dates = []
    curr = end_date
    while len(dates) < num_points:
        if curr.weekday() < 5:
            dates.append(curr)
        curr -= timedelta(days=1)
    dates.reverse()
    
    random.seed(hash(ticker) + 42)
    np.random.seed(abs(hash(ticker)) % (2**32))
    
    returns = np.random.normal(0.0002, 0.015, num_points)
    price_path = base * np.exp(np.cumsum(returns))
    
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    for p in price_path:
        pct_change = random.uniform(-0.02, 0.02)
        o = p * (1 - pct_change)
        c = p
        h = max(o, c) * random.uniform(1.0, 1.015)
        l = min(o, c) * random.uniform(0.985, 1.0)
        v = random.randint(5_000_000, 50_000_000)
        
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        volumes.append(v)
        
    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes
    }, index=pd.to_datetime(dates))
    
    return df


def get_market_ribbon() -> List[Dict[str, Any]]:
    """Get market index and trending quotes for the top marquee header bar."""
    indices = [
        {"symbol": "^GSPC", "label": "S&P 500", "price": 5117.09, "change_pct": 0.42},
        {"symbol": "^IXIC", "label": "NASDAQ", "price": 16288.36, "change_pct": 0.85},
        {"symbol": "GOOGL", "label": "GOOGL", "price": 178.35, "change_pct": 1.85},
        {"symbol": "NVDA", "label": "NVDA", "price": 128.40, "change_pct": 3.20},
        {"symbol": "AAPL", "label": "AAPL", "price": 185.63, "change_pct": 0.34},
        {"symbol": "MSFT", "label": "MSFT", "price": 420.15, "change_pct": 0.52},
    ]
    return indices

