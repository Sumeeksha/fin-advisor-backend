"""
Payload Builder — Assembles a compressed JSON payload for the LLM.
Strips chart history arrays and verbose fields to stay under ~600 tokens.
"""

from typing import Dict, Any, List, Optional


def build_llm_payload(
    ticker: str,
    quote: Dict[str, Any],
    indicators: Dict[str, Any],
    advice: Dict[str, Any],
    forecast: Dict[str, Any],
    news_items: List[Dict[str, Any]],
    risk_profile: str = "Moderate",
) -> Dict[str, Any]:
    """
    Build a compact context payload for the LLM containing ONLY key metrics.
    Designed to keep input under ~600 tokens.
    """
    payload: Dict[str, Any] = {
        "ticker": ticker,
        "risk_profile": risk_profile,
    }

    # ── Price snapshot ─────────────────────────
    payload["price"] = quote.get("price")
    payload["change_pct"] = quote.get("change_pct")

    # ── Algorithmic signal ─────────────────────
    payload["signal"] = advice.get("signal", "HOLD")
    payload["signal_confidence"] = advice.get("confidence", 0)
    payload["signal_score"] = f"{advice.get('score', 0)}/{advice.get('max_score', 0)}"
    payload["risk_factors"] = advice.get("risk_factors", [])[:3]

    # ── Technical indicators (scalars only) ────
    rsi = indicators.get("rsi", {})
    macd = indicators.get("macd", {})
    bb = indicators.get("bollinger_bands", {})
    ma = indicators.get("moving_averages", {})
    vol = indicators.get("volume", {})

    payload["rsi"] = rsi.get("current")
    payload["rsi_signal"] = rsi.get("signal")
    payload["macd_crossover"] = macd.get("crossover")
    payload["macd_histogram"] = macd.get("histogram")
    payload["bb_percent_b"] = bb.get("percent_b")
    payload["sma50"] = ma.get("sma_50")
    payload["sma200"] = ma.get("sma_200")
    payload["golden_cross"] = ma.get("golden_cross")
    payload["volume_relative"] = vol.get("relative")
    payload["trend"] = indicators.get("trend", "sideways")

    # ── Forecast snapshot ──────────────────────
    if "error" not in forecast:
        summary = forecast.get("summary", {})
        payload["forecast_method"] = forecast.get("method", "N/A")
        payload["forecast_direction"] = summary.get("direction")
        payload["forecast_change_pct"] = summary.get("projected_change_pct")
        payload["forecast_outlook"] = summary.get("outlook")

        # Include R² if available (linear regression quality)
        lr = forecast.get("linear_regression", {})
        if lr.get("r_squared") is not None:
            payload["lr_r_squared"] = lr["r_squared"]
    else:
        payload["forecast_method"] = "unavailable"
        payload["forecast_direction"] = None
        payload["forecast_change_pct"] = None
        payload["forecast_outlook"] = None

    # ── News (top 3-5, truncated headlines) ────
    headlines: List[str] = []
    for item in news_items[:5]:
        hl = item.get("headline", "")
        # Truncate to 80 chars to save tokens
        if len(hl) > 80:
            hl = hl[:77] + "..."
        if hl:
            headlines.append(hl)
    payload["news_headlines"] = headlines

    return payload
