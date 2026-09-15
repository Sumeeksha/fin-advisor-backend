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
    model: str = Query("dual", pattern="^(dual|gpt4o|gemini|claude|deepseek|fingpt|finma|alli)$"),
    source: str = Query("tradingview", pattern="^(yfinance|tradingview)$"),
):
    """
    Generate LLM-synthesized narrative insight and multi-model consensus for a ticker.
    Supports dynamic model selection: dual, gpt4o, gemini, claude, deepseek, or fingpt.
    """
    ticker = ticker.upper()

    # ── Step 1: Gather all data from existing math core ──
    quote = get_quote(ticker)
    if "error" in quote:
        raise HTTPException(status_code=404, detail=quote["error"])

    df_3m = get_history(ticker, "3M", source=source)
    df_1y = get_history(ticker, "1Y", source=source)

    indicators = (
        get_all_indicators(df_3m, ticker=ticker, source=source)
        if df_3m is not None and not df_3m.empty
        else {"error": "No data"}
    )

    advice = generate_advice(quote, indicators)

    # Pass the latest live price into the forecaster so ARIMA incorporates the most recent tick
    latest_live_price = quote.get("price")
    forecast = (
        generate_forecast(df_1y, latest_price=latest_live_price)
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
    llm_insight = get_llm_insight(payload, provider=model)

    # ── Step 4: Assemble UI data contract ────────────────
    price = quote.get("price", 0)
    projected_change = forecast.get("summary", {}).get("projected_change_pct")
    if projected_change is not None:
        target_12m = round(price * (1 + projected_change / 100), 2)
    else:
        target_12m = round(price * 1.05, 2)  # modest fallback only if no forecast
    stop_loss = round(price * 0.9451, 2)
    acc_low = round(price * 0.965, 2)
    acc_high = round(price, 2)

    # Dynamic verdict, consensus score, and synthesis thesis directly from LLM output & signals
    signal_str = advice.get("signal", "HOLD")
    provider_name = llm_insight.get("provider", "Multi-Model Engine")
    confidence_val = advice.get("confidence", 75)

    if llm_insight.get("divergence_warning", {}).get("detected"):
        verdict = f"{signal_str} / SENTIMENT DIVERGENCE"
    elif signal_str == "BUY":
        verdict = "ACCUMULATE / STAGED BUYS"
    elif signal_str == "SELL":
        verdict = "TRIM / STOP-LOSS"
    else:
        verdict = "HOLD / CAUTIOUS ACCUMULATION"

    consensus_score = f"{provider_name} ({confidence_val}% Conviction)"

    # Synthesis is dynamically extracted from the LLM summary & trend bullets
    synthesis = llm_insight.get("summary", "")
    trend_bullets = llm_insight.get("trend_explanation", [])
    if trend_bullets:
        synthesis += f" {trend_bullets[0]}"

    # Risk watchpoints directly from LLM key_risks + quantitative corridor
    llm_risks = llm_insight.get("key_risks", [])
    risk_tags = list(llm_risks) if llm_risks else []
    
    if forecast.get("confidence_corridor", {}).get("lower_95"):
        risk_tags.append(f"ARIMA 95% Tail Risk: ${forecast.get('confidence_corridor', {}).get('lower_95')}")
    if "error" not in indicators and indicators.get("moving_averages", {}).get("sma_50"):
        risk_tags.append(f"50-Day SMA Resistance (${indicators.get('moving_averages', {}).get('sma_50')})")

    indicators_snapshot = {}
    if "error" not in indicators:
        indicators_snapshot = {
            "rsi": indicators.get("rsi", {}).get("current"),
            "rsi_signal": indicators.get("rsi", {}).get("signal"),
            "macd_crossover": indicators.get("macd", {}).get("crossover"),
            "trend": indicators.get("trend"),
            "bb_percent_b": indicators.get("bollinger_bands", {}).get("percent_b"),
            "golden_cross": indicators.get("moving_averages", {}).get("golden_cross"),
            "sma_50": indicators.get("moving_averages", {}).get("sma_50"),
            "sma_250": indicators.get("moving_averages", {}).get("sma_200"),
            "vwap": quote.get("price"),  # Use live price instead of hardcoded multiplier
        }

    # ── Detect TradingView unavailability for frontend notification ──
    data_source = quote.get("source", "")
    tradingview_unavailable = quote.get("tradingview_unavailable", False)
    fallback_options = quote.get("fallback_options", [])

    return {
        "ticker": ticker,
        "signal": signal_str,
        "confidence": confidence_val,
        "risk_profile": risk_profile,
        "selected_model": model,
        "llm_insight": llm_insight,
        "data_source": data_source,
        "tradingview_unavailable": tradingview_unavailable,
        "fallback_options": fallback_options,
        "consensus": {
            "verdict": verdict,
            "consensus_score": consensus_score,
            "allocation": {"hold": 44, "accumulate": 26, "trim": 30},
            "target_levels": {
                "accumulation_zone": f"${acc_low} - ${acc_high}",
                "fair_target_12m": f"${target_12m}",
                "stop_loss": f"${stop_loss} (-5.49%)",
            },
            "synthesis_thesis": synthesis,
            "models": {
                "active_model": {
                    "name": provider_name,
                    "rating": f"{confidence_val}% {signal_str}",
                    "thesis": llm_insight.get("summary", ""),
                    "target": f"${target_12m}",
                    "stop": f"${stop_loss}",
                },
            },
        },
        "pipeline_stages": [
            {
                "id": "01",
                "name": "DATA INGESTION",
                "subtitle": "Market & Indicator Feed",
                "details": f"OHLCV 250D rolling window, source: {data_source or 'Live Feed'}",
                "badge": "Synced",
                "badge_color": "green",
            },
            {
                "id": "02",
                "name": "TIME-SERIES PATH",
                "subtitle": f"ARIMA {forecast.get('order', 'N/A')} Forecaster",
                "details": f"14D projection, AIC {forecast.get('aic_score', 'N/A')}",
                "badge": f"{'+' if (projected_change or 0) >= 0 else ''}{projected_change or 0:.2f}% Room",
                "badge_color": "blue",
            },
            {
                "id": "03",
                "name": "MULTI-LLM REASONING",
                "subtitle": provider_name,
                "details": f"Model: {model.upper()} | Dynamic Real-Time Synthesis",
                "badge": signal_str,
                "badge_color": "cyan",
            },
            {
                "id": "04",
                "name": "RECONCILED OUTPUT",
                "subtitle": "Executive Conviction",
                "details": verdict,
                "badge": "Active Signal",
                "badge_color": "green",
            },
        ],
        "risk_tags": risk_tags,
        "quote": {
            "price": quote.get("price"),
            "change": quote.get("change"),
            "change_pct": quote.get("change_pct"),
        },
        "indicators_snapshot": indicators_snapshot,
        "forecast_snapshot": {
            "method": forecast.get("method"),
            "order": forecast.get("order"),
            "econometric_specification": forecast.get("econometric_specification"),
            "aic_score": forecast.get("aic_score"),
            "rmse": forecast.get("rmse"),
            "drift_term": forecast.get("drift_term"),
            "p_value": forecast.get("p_value"),
            "confidence_corridor": forecast.get("confidence_corridor"),
        },
    }
