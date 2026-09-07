"""
News Service — Unified intelligence feed aggregator.

Sources (in priority order):
  1. SEC EDGAR REST API  — regulatory filings (10-K, 10-Q, 8-K) — FREE, no key
  2. Finnhub             — company news (uses existing FINNHUB_API_KEY)
  3. yfinance            — news fallback
  4. RSS (feedparser)    — Google/Yahoo Finance RSS fallback

All sources normalise to a single NewsItem schema with a canonical URL.
Results are deduped by URL, sorted newest-first, and cached in-memory.

Cache TTL:
  - News (Finnhub / yfinance / RSS) : 5 minutes
  - SEC filings                     : 30 minutes
"""

import os
import time
import logging
import hashlib
import calendar
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)

# ── Keys (read from env, same pattern as data_provider) ──────────────────────
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")

# ── SEC EDGAR — required User-Agent (SEC blocks anonymous requests) ──────────
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "FinancialAdvisoryApp/1.0 help.finadvisor@gmail.com",
)
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# ── In-memory cache ──────────────────────────────────────────────────────────
_news_cache: Dict[str, Dict] = {}          # key -> {"ts": float, "data": list}
_sec_cache:  Dict[str, Dict] = {}          # key -> {"ts": float, "data": list}
_cik_cache:  Dict[str, Optional[str]] = {} # ticker -> CIK string (or None)

NEWS_TTL = 300   # 5 minutes
SEC_TTL  = 1800  # 30 minutes


# ── Standardised schema ───────────────────────────────────────────────────────
@dataclass
class NewsItem:
    id:           str
    type:         str           # "filing" | "news" | "rss"
    title:        str
    summary:      str
    source:       str
    source_logo:  str
    published_at: str           # ISO 8601 UTC
    published_ts: int           # Unix timestamp
    url:          str           # canonical direct link — never empty
    image:        str
    category:     str           # e.g. "8-K", "earnings", "market"
    sentiment:    Optional[str] # "positive" | "negative" | "neutral" | None


def _make_id(prefix: str, url: str) -> str:
    """Short stable ID from URL hash."""
    return f"{prefix}_{hashlib.md5(url.encode()).hexdigest()[:10]}"


def _ts_to_iso(ts: int) -> str:
    """Convert Unix timestamp to ISO 8601 UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_to_ts(iso: str) -> int:
    """Parse ISO 8601 string to Unix timestamp."""
    try:
        iso = iso.replace("Z", "+00:00")
        return int(datetime.fromisoformat(iso).timestamp())
    except Exception:
        return int(time.time())


# ═══════════════════════════════════════════════════════════════════════════════
# SEC EDGAR — Regulatory Filings
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_cik(ticker: str) -> Optional[str]:
    """
    Resolve ticker -> 10-digit zero-padded CIK via SEC EDGAR company_tickers.json.
    Result is cached indefinitely (CIK never changes for a company).
    """
    ticker_upper = ticker.upper()
    if ticker_upper in _cik_cache:
        return _cik_cache[ticker_upper]

    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=8)
        if resp.status_code == 200:
            for entry in resp.json().values():
                if entry.get("ticker", "").upper() == ticker_upper:
                    cik_padded = str(entry["cik_str"]).zfill(10)
                    _cik_cache[ticker_upper] = cik_padded
                    logger.info(f"SEC CIK resolved: {ticker_upper} -> {cik_padded}")
                    return cik_padded
    except Exception as e:
        logger.warning(f"SEC CIK lookup failed for {ticker_upper}: {e}")

    _cik_cache[ticker_upper] = None
    return None


def _filing_url(cik: str, accession_number: str, primary_doc: str = "") -> str:
    """Build the direct SEC filing URL from CIK + accession number."""
    acc_clean = accession_number.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/"
    if primary_doc:
        return base + primary_doc
    return base + f"{accession_number}-index.htm"


def _fetch_sec_filings(
    ticker: str,
    form_types: List[str] = None,
    limit: int = 10,
) -> List[NewsItem]:
    """
    Fetch recent SEC EDGAR regulatory filings for a ticker.

    Official free SEC EDGAR REST API endpoint:
      GET https://data.sec.gov/submissions/CIK{cik}.json

    User-Agent header is mandatory per SEC policy.
    Form types supported: 10-K, 10-Q, 8-K, DEF 14A, SC 13G
    """
    if form_types is None:
        form_types = ["10-K", "10-Q", "8-K"]

    cache_key = f"{ticker}_{'_'.join(sorted(form_types))}"
    cached = _sec_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < SEC_TTL:
        return cached["data"]

    items: List[NewsItem] = []
    cik = _resolve_cik(ticker)
    if not cik:
        logger.warning(f"No SEC CIK found for {ticker}, skipping SEC filings.")
        _sec_cache[cache_key] = {"ts": time.time(), "data": items}
        return items

    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"SEC submissions fetch failed: HTTP {resp.status_code}")
            return items

        data = resp.json()
        filings = data.get("filings", {}).get("recent", {})

        forms         = filings.get("form", [])
        filed_dates   = filings.get("filingDate", [])
        accessions    = filings.get("accessionNumber", [])
        descriptions  = filings.get("primaryDocDescription", [])
        primary_docs  = filings.get("primaryDocument", [])
        company_name  = data.get("name", ticker)

        collected = 0
        for i, form in enumerate(forms):
            if collected >= limit:
                break
            if form not in form_types:
                continue

            acc          = accessions[i]   if i < len(accessions)   else ""
            filed        = filed_dates[i]  if i < len(filed_dates)  else ""
            desc         = descriptions[i] if i < len(descriptions) else ""
            primary_doc  = primary_docs[i] if i < len(primary_docs) else ""

            direct_url = _filing_url(cik, acc, primary_doc)

            # Parse filing date
            if filed:
                try:
                    filed_ts  = int(datetime.strptime(filed, "%Y-%m-%d")
                                    .replace(tzinfo=timezone.utc).timestamp())
                    filed_iso = f"{filed}T00:00:00Z"
                except Exception:
                    filed_ts  = int(time.time())
                    filed_iso = _ts_to_iso(filed_ts)
            else:
                filed_ts  = int(time.time())
                filed_iso = _ts_to_iso(filed_ts)

            title = (
                f"{company_name} — {form}: {desc} ({filed})"
                if desc else
                f"{company_name} — {form} ({filed})"
            )
            summary = (
                f"Regulatory filing submitted to the SEC on {filed}. "
                f"Form type: {form}. "
                f"Click to read the full official disclosure on SEC EDGAR."
            )

            items.append(NewsItem(
                id=_make_id(f"sec_{form}", direct_url),
                type="filing",
                title=title,
                summary=summary,
                source="SEC EDGAR",
                source_logo="https://www.sec.gov/files/sec-logo.png",
                published_at=filed_iso,
                published_ts=filed_ts,
                url=direct_url,
                image="",
                category=form,
                sentiment=None,
            ))
            collected += 1

    except Exception as e:
        logger.error(f"SEC EDGAR fetch error for {ticker}: {e}")

    _sec_cache[cache_key] = {"ts": time.time(), "data": items}
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# Finnhub — Company News
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_finnhub_news(ticker: str, limit: int = 10) -> List[NewsItem]:
    """Fetch company news from Finnhub (uses existing FINNHUB_API_KEY env var)."""
    if not FINNHUB_KEY or FINNHUB_KEY in ("", "YOUR_FINNHUB_KEY_HERE"):
        return []

    cache_key = f"finnhub_{ticker}"
    cached = _news_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < NEWS_TTL:
        return cached["data"]

    items: List[NewsItem] = []
    try:
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        to_date   = datetime.now().strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={from_date}&to={to_date}&token={FINNHUB_KEY}"
        )
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            for raw in resp.json()[:limit]:
                article_url = raw.get("url", "")
                if not article_url:
                    continue
                ts = raw.get("datetime", int(time.time()))
                items.append(NewsItem(
                    id=_make_id("finnhub", article_url),
                    type="news",
                    title=raw.get("headline", ""),
                    summary=raw.get("summary", ""),
                    source=raw.get("source", "Finnhub"),
                    source_logo="",
                    published_at=_ts_to_iso(ts),
                    published_ts=ts,
                    url=article_url,
                    image=raw.get("image", ""),
                    category=raw.get("category", "market"),
                    sentiment=None,
                ))
    except Exception as e:
        logger.warning(f"Finnhub news fetch error for {ticker}: {e}")

    _news_cache[cache_key] = {"ts": time.time(), "data": items}
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# yfinance — News Fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_yfinance_news(ticker: str, limit: int = 10) -> List[NewsItem]:
    """Fetch news via yfinance (no API key needed)."""
    cache_key = f"yf_{ticker}"
    cached = _news_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < NEWS_TTL:
        return cached["data"]

    items: List[NewsItem] = []
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        for raw in (tk.news or [])[:limit]:
            content     = raw.get("content", {})
            article_url = (
                content.get("canonicalUrl", {}).get("url")
                or raw.get("link", "")
            )
            if not article_url:
                continue

            title   = content.get("title") or raw.get("title", "")
            summary = content.get("summary", "")
            source  = (
                content.get("provider", {}).get("displayName")
                or raw.get("publisher", "Yahoo Finance")
            )
            pub_date = content.get("pubDate", "")
            ts = _iso_to_ts(pub_date) if pub_date else int(time.time())

            items.append(NewsItem(
                id=_make_id("yf", article_url),
                type="news",
                title=title,
                summary=summary,
                source=source,
                source_logo="",
                published_at=_ts_to_iso(ts),
                published_ts=ts,
                url=article_url,
                image="",
                category="market",
                sentiment=None,
            ))
    except Exception as e:
        logger.warning(f"yfinance news fetch error for {ticker}: {e}")

    _news_cache[cache_key] = {"ts": time.time(), "data": items}
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# RSS — Google Finance / Yahoo Finance Feeds (feedparser)
# ═══════════════════════════════════════════════════════════════════════════════

RSS_FEEDS_TEMPLATE = [
    (
        "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
        "Google Finance",
    ),
    (
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        "Yahoo Finance",
    ),
]


def _fetch_rss_news(ticker: str, limit: int = 10) -> List[NewsItem]:
    """
    Fetch news via RSS feeds using feedparser.
    Falls back gracefully if feedparser is not installed.
    """
    cache_key = f"rss_{ticker}"
    cached = _news_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < NEWS_TTL:
        return cached["data"]

    items: List[NewsItem] = []

    try:
        import feedparser
    except ImportError:
        logger.warning(
            "feedparser not installed — RSS feeds unavailable. "
            "Install with: pip install feedparser==6.0.11"
        )
        _news_cache[cache_key] = {"ts": time.time(), "data": items}
        return items

    for feed_url_template, source_name in RSS_FEEDS_TEMPLATE:
        if len(items) >= limit:
            break
        feed_url = feed_url_template.format(ticker=ticker)
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                if len(items) >= limit:
                    break
                article_url = entry.get("link", "")
                if not article_url:
                    continue

                title   = entry.get("title", "")
                summary = entry.get("summary", "") or entry.get("description", "")

                published_parsed = entry.get("published_parsed")
                ts = calendar.timegm(published_parsed) if published_parsed else int(time.time())

                items.append(NewsItem(
                    id=_make_id("rss", article_url),
                    type="rss",
                    title=title,
                    summary=summary,
                    source=source_name,
                    source_logo="",
                    published_at=_ts_to_iso(ts),
                    published_ts=ts,
                    url=article_url,
                    image="",
                    category="market",
                    sentiment=None,
                ))
        except Exception as e:
            logger.warning(f"RSS feed error ({source_name}) for {ticker}: {e}")

    _news_cache[cache_key] = {"ts": time.time(), "data": items}
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_all_news(
    ticker: str,
    types: Optional[List[str]] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Aggregate news and regulatory filings for a ticker from all configured sources.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL")
        types:  Source types to include — any of ["news", "sec", "rss"].
                Defaults to all three.
        limit:  Maximum total items to return (default 20).

    Returns:
        List of NewsItem dicts, sorted newest-first, deduplicated by URL.
        Every item is guaranteed to have a non-empty canonical URL.
    """
    ticker = ticker.upper().strip()
    if types is None:
        types = ["news", "sec", "rss"]

    all_items: List[NewsItem] = []

    # 1. SEC EDGAR Filings
    if "sec" in types:
        sec_items = _fetch_sec_filings(ticker, form_types=["10-K", "10-Q", "8-K"], limit=10)
        all_items.extend(sec_items)
        logger.info(f"[{ticker}] SEC: {len(sec_items)} filings")

    # 2. Finnhub news (primary)
    if "news" in types:
        finnhub_items = _fetch_finnhub_news(ticker, limit=limit)
        all_items.extend(finnhub_items)
        logger.info(f"[{ticker}] Finnhub: {len(finnhub_items)} articles")

        # yfinance fallback when Finnhub returns nothing
        if not finnhub_items:
            yf_items = _fetch_yfinance_news(ticker, limit=limit)
            all_items.extend(yf_items)
            logger.info(f"[{ticker}] yfinance fallback: {len(yf_items)} articles")

    # 3. RSS feeds
    if "rss" in types:
        rss_items = _fetch_rss_news(ticker, limit=10)
        all_items.extend(rss_items)
        logger.info(f"[{ticker}] RSS: {len(rss_items)} articles")

    # Deduplicate by URL
    seen_urls: set = set()
    unique: List[NewsItem] = []
    for item in all_items:
        if item.url and item.url not in seen_urls:
            seen_urls.add(item.url)
            unique.append(item)

    # Sort newest-first and apply limit
    unique.sort(key=lambda x: x.published_ts, reverse=True)
    result = [asdict(item) for item in unique[:limit]]

    logger.info(f"[{ticker}] Total returned: {len(result)} items (types={types})")
    return result
