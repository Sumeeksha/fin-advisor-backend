"""Insight router — /api/insight/{ticker}
Orchestrates the full pipeline: Math Core → Payload Assembly → LLM → UI Output.
"""

from fastapi import APIRouter, HTTPException, Query
from services.data_provider import get_quote, get_history, get_news
from services.indicators import get_all_indicators
from services.advisor import generate_advice
from services.forecaster import generate_forecast
from services.payload_builder import build_llm_payload
from services.llm_client import get_llm_insight

router = APIRouter()


@router.get("/{ticker}")
def insight(
    ticker: str,
    risk_profile: str = Query("Moderate", pattern="^(Conservative|Moderate|Aggressive)$"),
):
    """
    Generate LLM-synthesized narrative insight for a ticker.
    Combines forecasts, indicators, advice, and news into a concise analysis.
    """
    ticker = ticker.upper()

    # ── Step 1: Gather all data from existing math core ──
    quote = get_quote(ticker)
    if "error" in quote:
        raise HTTPException(status_code=404, detail=quote["error"])

    df_3m = get_history(ticker, "3M")
    df_1y = get_history(ticker, "1Y")

    indicators = (
        get_all_indicators(df_3m)
        if df_3m is not None and not df_3m.empty
        else {"error": "No data"}
    )

    advice = generate_advice(quote, indicators)

    forecast = (
        generate_forecast(df_1y)
        if df_1y is not None and not df_1y.empty
        else {"error": "No data"}
    )

    news_items = get_news(ticker)

    # ── Step 2: Build compressed LLM payload ─────────────
    payload = build_llm_payload(
        ticker=ticker,
        quote=quote,
        indicators=indicators,
        advice=advice,
        forecast=forecast,
        news_items=news_items,
        risk_profile=risk_profile,
    )

    # ── Step 3: Get LLM insight (or fallback) ────────────
    llm_insight = get_llm_insight(payload)

    # ── Step 4: Assemble UI data contract ────────────────
    # Compact snapshots for frontend context (no chart arrays)
    indicators_snapshot = {}
    if "error" not in indicators:
        indicators_snapshot = {
            "rsi": indicators.get("rsi", {}).get("current"),
            "rsi_signal": indicators.get("rsi", {}).get("signal"),
            "macd_crossover": indicators.get("macd", {}).get("crossover"),
            "trend": indicators.get("trend"),
            "bb_percent_b": indicators.get("bollinger_bands", {}).get("percent_b"),
            "golden_cross": indicators.get("moving_averages", {}).get("golden_cross"),
        }

    forecast_snapshot = {}
    if "error" not in forecast:
        summary = forecast.get("summary", {})
        forecast_snapshot = {
            "method": forecast.get("method"),
            "direction": summary.get("direction"),
            "projected_change_pct": summary.get("projected_change_pct"),
            "outlook": summary.get("outlook"),
        }

    return {
        "ticker": ticker,
        "signal": advice.get("signal", "HOLD"),
        "confidence": advice.get("confidence", 0),
        "risk_profile": risk_profile,
        "llm_insight": llm_insight,
        "quote": {
            "price": quote.get("price"),
            "change": quote.get("change"),
            "change_pct": quote.get("change_pct"),
        },
        "indicators_snapshot": indicators_snapshot,
        "forecast_snapshot": forecast_snapshot,
    }
