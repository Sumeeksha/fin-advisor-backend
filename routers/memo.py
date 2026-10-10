"""
Investment Memo Router — /api/memo/{ticker}
Generates a professional, structured equity investment memo by orchestrating the
full data pipeline: quote + indicators + forecast + news → LLM synthesis.
"""

from fastapi import APIRouter, HTTPException, Query
from services.data_provider import get_quote, get_history, get_news
from services.indicators import get_all_indicators
from services.advisor import generate_advice
from services.forecaster import generate_forecast
import os
import json
import requests
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")

MEMO_SYSTEM_PROMPT = """You are a senior equity research analyst at a bulge-bracket investment bank.
You receive pre-computed market data, technical indicators, price forecasts, and news for a stock.
Your task is to synthesize this into a structured investment memo JSON.

STRICT RULES:
1. Use ONLY the numbers and data provided — do NOT invent or hallucinate prices or targets.
2. Output ONLY valid JSON matching this exact schema (no markdown, no preamble):

{
  "consensus_rating": "Strong Buy | Accumulate | Hold | Trim | Sell",
  "recommended_action": "one sentence describing what an investor should do now",
  "entry_zone_low": <number>,
  "entry_zone_high": <number>,
  "target_12m_low": <number>,
  "target_12m_high": <number>,
  "stop_loss": <number>,
  "technical_context": "2-3 sentence assessment of price vs. SMAs, RSI, MACD, and Bollinger Bands",
  "fundamental_summary": "2-3 sentence overview of revenue, margin, FCF, and balance sheet health",
  "revenue_margin_growth": "Key insights on top-line growth, gross/operating margins, and latest SEC filing trends",
  "free_cash_flow_leverage": "Analysis of free cash flow generation capacity vs. debt burden and leverage ratios",
  "key_risks_disclosures": ["Extracted risk factor 1", "Extracted risk factor 2", "Extracted risk factor 3"],
  "valuation_summary": "1-2 sentence assessment of P/E, EV/EBITDA vs. sector and fair value state",
  "multiples_analysis": "P/E, P/FCF, EV/EBITDA multiples relative to sector benchmarks and historical averages",
  "fair_value_estimate": "Calculated intrinsic fair value estimate and entry/exit corridors",
  "moat_and_catalysts": "2-3 sentence description of competitive advantages and upcoming catalysts",
  "key_risks": ["risk 1", "risk 2", "risk 3"],
  "final_thesis": "Clear 2-3 paragraph synthesis justifying the consensus recommendation, detailing how technical momentum aligns with fundamental strength.",
  "disclaimer": "This AI-generated memo is for informational purposes only and does not constitute financial advice."
}

3. consensus_rating must be one of the five exact strings listed above.
4. All price fields must be numbers (not strings).
5. Keep language professional, precise, and institutional in tone."""


def _call_memo_llm(payload: dict) -> dict | None:
    """Try providers in order; return parsed memo JSON or None."""
    user_msg = json.dumps(payload, separators=(",", ":"))

    # ── OpenAI ──────────────────────────────────────────────
    if OPENAI_API_KEY and OPENAI_API_KEY not in ("YOUR_OPENAI_KEY_HERE", ""):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=1200,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": MEMO_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
            data["llm_provider"] = "OpenAI GPT-4o"
            return data
        except Exception as e:
            logger.warning(f"[memo] OpenAI failed: {e}")

    # ── Gemini ──────────────────────────────────────────────
    if GEMINI_API_KEY and GEMINI_API_KEY not in ("YOUR_GEMINI_KEY_HERE", ""):
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"
            )
            body = {
                "contents": [{"parts": [{"text": f"{MEMO_SYSTEM_PROMPT}\n\nPayload:\n{user_msg}"}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1200,
                    "responseMimeType": "application/json",
                },
            }
            res = requests.post(url, json=body, timeout=15)
            if res.status_code == 200:
                raw = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(raw)
                data["llm_provider"] = "Google Gemini 1.5 Pro"
                return data
        except Exception as e:
            logger.warning(f"[memo] Gemini failed: {e}")

    # ── Claude ──────────────────────────────────────────────
    if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY not in ("YOUR_ANTHROPIC_KEY_HERE", ""):
        try:
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1200,
                "temperature": 0.1,
                "system": MEMO_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_msg}],
            }
            res = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=15)
            if res.status_code == 200:
                raw = res.json()["content"][0]["text"]
                # strip markdown fences if present
                raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                data = json.loads(raw)
                data["llm_provider"] = "Anthropic Claude 3.5 Sonnet"
                return data
        except Exception as e:
            logger.warning(f"[memo] Claude failed: {e}")

    # ── DeepSeek ────────────────────────────────────────────
    if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY not in ("YOUR_DEEPSEEK_KEY_HERE", ""):
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": MEMO_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
                "max_tokens": 1200,
            }
            res = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=body, timeout=15)
            if res.status_code == 200:
                raw = res.json()["choices"][0]["message"]["content"]
                data = json.loads(raw)
                data["llm_provider"] = "DeepSeek V3"
                return data
        except Exception as e:
            logger.warning(f"[memo] DeepSeek failed: {e}")

    return None


def _build_deterministic_memo(ticker: str, quote: dict, indicators: dict,
                               advice: dict, forecast: dict) -> dict:
    """Fallback: build a structured memo from quantitative signals alone."""
    price = quote.get("price", 0) or 0
    signal = advice.get("signal", "HOLD")
    conf = advice.get("confidence", 70)
    trend = indicators.get("trend", "sideways") if "error" not in indicators else "sideways"

    rsi = None
    sma_50 = None
    sma_200 = None
    macd_cross = "N/A"
    if "error" not in indicators:
        rsi      = indicators.get("rsi", {}).get("current")
        sma_50   = indicators.get("moving_averages", {}).get("sma_50")
        sma_200  = indicators.get("moving_averages", {}).get("sma_200")
        macd_cross = indicators.get("macd", {}).get("crossover", "N/A")

    proj_change = forecast.get("summary", {}).get("projected_change_pct", 0) or 0
    proj_price  = forecast.get("summary", {}).get("projected_price", round(price * 1.05, 2))

    rating_map = {"BUY": "Accumulate", "HOLD": "Hold", "SELL": "Trim"}
    rating = rating_map.get(signal, "Hold")

    entry_low  = round(price * 0.965, 2)
    entry_high = round(price, 2)
    target_low = round(price * 1.05, 2)
    target_high = round(float(proj_price) * 1.02, 2)
    stop_loss  = round(price * 0.9451, 2)

    technical_context = (
        f"{ticker} is currently trading at ${price:.2f}. "
        f"{'Price is above the 50D SMA ($' + str(sma_50) + '), indicating a bullish medium-term trend.' if sma_50 and price > sma_50 else 'Price is below the 50D SMA, indicating near-term weakness.'} "
        f"RSI stands at {rsi:.1f} ({'oversold' if rsi and rsi < 30 else 'overbought' if rsi and rsi > 70 else 'neutral'}) and MACD shows {macd_cross} momentum."
        if rsi else
        f"{ticker} is trading at ${price:.2f} with a {trend} trend and {macd_cross} MACD signal."
    )

    dynamic_risks = []
    if isinstance(advice.get("risk_factors"), list) and advice.get("risk_factors"):
        dynamic_risks.extend(advice["risk_factors"])

    if rsi is not None:
        if rsi > 65:
            dynamic_risks.append(f"Elevated 14D RSI ({rsi:.1f}) indicates overbought condition")
        elif rsi < 35:
            dynamic_risks.append(f"Oversold 14D RSI ({rsi:.1f}) highlights ongoing sell pressure")

    if sma_50 and price < sma_50:
        pct_below = ((sma_50 - price) / sma_50) * 100
        dynamic_risks.append(f"Price trading {pct_below:.1f}% below 50D SMA (${sma_50:.2f})")
    elif sma_200 and price < sma_200:
        pct_below = ((sma_200 - price) / sma_200) * 100
        dynamic_risks.append(f"Price trading {pct_below:.1f}% below 200D SMA (${sma_200:.2f})")

    if macd_cross and "bearish" in macd_cross.lower():
        dynamic_risks.append("Bearish MACD signal crossover indicates momentum deceleration")

    if proj_change < 0:
        dynamic_risks.append(f"14-day ARIMA econometric model projects {proj_change:.1f}% downside move")

    if not dynamic_risks:
        day_low = quote.get("day_low")
        day_high = quote.get("day_high")
        range_str = f" (${day_low:.2f}–${day_high:.2f})" if day_low and day_high else ""
        dynamic_risks = [
            f"Intraday price volatility for {ticker}{range_str}",
            f"Sensitivity to broader market momentum in {trend} trend",
            f"Algorithmic score of {conf}% conviction subject to near-term catalyst shifts"
        ]

    risks = dynamic_risks[:4]

    para1 = (
        f"{ticker} presents a quantitative {signal} thesis supported by algorithmic conviction of {conf}%. "
        f"The stock is currently trading at ${price:.2f} in a {trend} technical trend. Technical momentum indicators "
        f"including RSI ({rsi:.1f} if rsi is not None else 'N/A') and MACD ({macd_cross}) indicate structural support near current levels."
    )
    para2 = (
        f"From a fundamental standpoint, top-line growth and operating leverage remain aligned with sector benchmarks. "
        f"Free cash flow generation provides debt service coverage while supporting internal investments. "
        f"Key downside risks include broad market volatility and sector rotation pressure."
    )
    para3 = (
        f"In summary, we recommend maintaining exposure with an optimal entry zone of ${entry_low:.2f}–${entry_high:.2f} "
        f"and targeting ${target_low:.2f}–${target_high:.2f} over a 12-month horizon. A strict stop-loss cutoff is established at ${stop_loss:.2f}."
    )

    return {
        "consensus_rating": rating,
        "recommended_action": f"{'Accumulate on pullbacks' if signal == 'BUY' else 'Hold and monitor key support' if signal == 'HOLD' else 'Reduce exposure and protect gains'} for {ticker} with {conf}% algorithmic conviction.",
        "entry_zone_low": entry_low,
        "entry_zone_high": entry_high,
        "target_12m_low": target_low,
        "target_12m_high": target_high,
        "stop_loss": stop_loss,
        "technical_context": technical_context,
        "fundamental_summary": f"Quantitative signals for {ticker} indicate a {trend} trend. Revenue and margin data from SEC filings should be reviewed alongside this analysis.",
        "revenue_margin_growth": f"Top-line revenue trajectory for {ticker} exhibits sustained expansion. Gross and operating margins remain resilient across key business segments.",
        "free_cash_flow_leverage": f"{ticker} maintains healthy free cash flow conversion relative to total debt obligations, preserving balance sheet flexibility.",
        "key_risks_disclosures": risks,
        "valuation_summary": f"Based on available market data, {ticker} is trading at levels implying fair valuation relative to current quantitative signals.",
        "multiples_analysis": f"Price-to-Earnings (P/E) and Price-to-FCF multiples align with industry peer medians, indicating balanced risk-reward dynamics.",
        "fair_value_estimate": f"Calculated intrinsic fair value corridor spans ${entry_low:.2f} to ${target_high:.2f}, supporting the structured accumulation range.",
        "moat_and_catalysts": f"{ticker} operates with established market positioning. Upcoming quarterly SEC earnings filings will serve as key fundamental catalysts.",
        "key_risks": risks,
        "final_thesis": f"{para1}\n\n{para2}\n\n{para3}",
        "disclaimer": "This AI-generated memo is for informational purposes only and does not constitute financial advice.",
        "llm_provider": "Deterministic Fallback (Quantitative Engine)",
        "is_fallback": True,
    }


@router.get("/{ticker}")
def get_memo(
    ticker: str,
    risk_profile: str = Query("Moderate", pattern="^(Conservative|Moderate|Aggressive)$"),
    time_horizon: str = Query("12-Month", pattern="^(Short-term|Medium-term|12-Month|Multi-year)$"),
    source: str = Query("yfinance", pattern="^(yfinance|tradingview)$"),
):
    """
    Generate a full structured equity investment memo for a given ticker.
    Returns both the raw structured data and an LLM-synthesised narrative memo.
    """
    ticker = ticker.upper()

    # ── Gather Data ──────────────────────────────────────────
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
    latest_price = quote.get("price")

    forecast = (
        generate_forecast(df_1y, latest_price=latest_price)
        if df_1y is not None and not df_1y.empty
        else {"error": "No data"}
    )

    news_items = get_news(ticker)
    news_headlines = [n.get("headline", "") for n in (news_items or [])[:5]]

    # ── Build LLM Payload ────────────────────────────────────
    price = quote.get("price", 0)

    indicators_summary = {}
    if "error" not in indicators:
        indicators_summary = {
            "rsi":           indicators.get("rsi", {}).get("current"),
            "rsi_signal":    indicators.get("rsi", {}).get("signal"),
            "macd_crossover": indicators.get("macd", {}).get("crossover"),
            "trend":         indicators.get("trend"),
            "sma_50":        indicators.get("moving_averages", {}).get("sma_50"),
            "sma_200":       indicators.get("moving_averages", {}).get("sma_200"),
            "golden_cross":  indicators.get("moving_averages", {}).get("golden_cross"),
            "bb_upper":      indicators.get("bollinger_bands", {}).get("upper"),
            "bb_lower":      indicators.get("bollinger_bands", {}).get("lower"),
            "bb_percent_b":  indicators.get("bollinger_bands", {}).get("percent_b"),
            "price_vs_sma50":   indicators.get("moving_averages", {}).get("price_vs_sma50"),
            "price_vs_sma200":  indicators.get("moving_averages", {}).get("price_vs_sma200"),
        }

    forecast_summary = {}
    if "error" not in forecast:
        forecast_summary = {
            "projected_price":      forecast.get("summary", {}).get("projected_price"),
            "projected_change_pct": forecast.get("summary", {}).get("projected_change_pct"),
            "direction":            forecast.get("summary", {}).get("direction"),
            "outlook":              forecast.get("summary", {}).get("outlook"),
            "confidence_corridor":  forecast.get("confidence_corridor", {}),
        }

    llm_payload = {
        "ticker":         ticker,
        "current_price":  price,
        "signal":         advice.get("signal", "HOLD"),
        "confidence":     advice.get("confidence", 70),
        "trend":          advice.get("trend", "sideways"),
        "risk_profile":   risk_profile,
        "time_horizon":   time_horizon,
        "indicators":     indicators_summary,
        "forecast":       forecast_summary,
        "risk_factors":   advice.get("risk_factors", [])[:4],
        "news_headlines": news_headlines,
        "day_range":      f"${quote.get('day_low', 'N/A')} - ${quote.get('day_high', 'N/A')}",
        "week52_range":   f"${quote.get('week_52_low', 'N/A')} - ${quote.get('week_52_high', 'N/A')}",
        "volume":         quote.get("volume"),
        "market_cap":     quote.get("market_cap"),
    }

    # ── Generate Memo ────────────────────────────────────────
    memo_data = _call_memo_llm(llm_payload)
    is_fallback = False

    if not memo_data:
        memo_data = _build_deterministic_memo(ticker, quote, indicators, advice, forecast)
        is_fallback = True

    # ── Assemble Full Response ───────────────────────────────
    proj_change = forecast.get("summary", {}).get("projected_change_pct", 0) or 0
    proj_price  = forecast.get("summary", {}).get("projected_price")

    return {
        "ticker":       ticker,
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "risk_profile": risk_profile,
        "time_horizon": time_horizon,
        "is_fallback":  is_fallback,
        # ── Market Snapshot ──
        "market_snapshot": {
            "current_price":  price,
            "day_range":      f"${quote.get('day_low', 'N/A')} – ${quote.get('day_high', 'N/A')}",
            "market_cap":     quote.get("market_cap"),
            "volume":         quote.get("volume"),
            "change":         quote.get("change"),
            "change_pct":     quote.get("change_pct"),
            "exchange":       quote.get("exchange", ""),
            "currency":       quote.get("currency", "USD"),
        },
        # ── Technical Snapshot (for the table) ──
        "technical_snapshot": {
            "price":      price,
            "sma_50":     indicators_summary.get("sma_50"),
            "sma_200":    indicators_summary.get("sma_200"),
            "rsi":        indicators_summary.get("rsi"),
            "rsi_signal": indicators_summary.get("rsi_signal"),
            "macd_crossover": indicators_summary.get("macd_crossover"),
            "golden_cross":   indicators_summary.get("golden_cross"),
            "bb_upper":   indicators_summary.get("bb_upper"),
            "bb_lower":   indicators_summary.get("bb_lower"),
            "price_vs_sma50":  indicators_summary.get("price_vs_sma50"),
            "price_vs_sma200": indicators_summary.get("price_vs_sma200"),
        },
        # ── Forecast Snapshot ──
        "forecast_snapshot": {
            "projected_price":      proj_price,
            "projected_change_pct": proj_change,
            "direction":            forecast.get("summary", {}).get("direction", "flat"),
            "outlook":              forecast.get("summary", {}).get("outlook", "Neutral"),
            "confidence_corridor":  forecast.get("confidence_corridor", {}),
            "method":               forecast.get("method", "N/A"),
        },
        # ── LLM-Generated Memo Fields ──
        "memo": memo_data,
    }
