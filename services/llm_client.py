"""
LLM Client — Thin OpenAI API wrapper for narrative synthesis.
Token-optimized: gpt-4o, temperature=0.1, max_tokens=350, JSON output mode.
Graceful fallback when API key is absent or call fails.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = "gpt-4o"
TEMPERATURE = 0.1
MAX_TOKENS = 350
MAX_RETRIES = 1
RETRY_DELAY_S = 2.0

SYSTEM_PROMPT = """You are a concise financial synthesis assistant. You receive pre-computed algorithmic signals, technical indicators, price forecasts, and news headlines for a stock.

RULES — follow these strictly:
1. Do NOT recalculate stock prices or generate your own price targets. Use only the numbers provided.
2. Keep all answers brief and direct. Use concise bullet points. No fluff, boilerplate, or filler.
3. Check for conflicts between the algorithmic signal and news sentiment. Flag any divergence.
4. Output ONLY valid JSON matching this exact schema:
{
  "summary": "1-2 sentence overview of the stock's current position and outlook",
  "trend_explanation": ["bullet 1 explaining ARIMA/regression trend", "bullet 2 (optional)"],
  "divergence_warning": {"detected": bool, "explanation": "brief text or empty string"},
  "key_risks": ["risk 1", "risk 2", "risk 3 (optional)"],
  "disclaimer": "This AI-generated synthesis is for informational purposes only and does not constitute financial advice. Consult a licensed advisor before investing."
}
5. Tailor risk tolerance language to the user's risk profile (Conservative/Moderate/Aggressive).
6. Keep total response under 300 tokens."""


def _is_api_available() -> bool:
    """Check if OpenAI API key is configured."""
    return bool(OPENAI_API_KEY) and OPENAI_API_KEY != "YOUR_OPENAI_KEY_HERE"


def _build_fallback_insight(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback when LLM is unavailable."""
    signal = payload.get("signal", "HOLD")
    direction = payload.get("forecast_direction", "flat")
    change_pct = payload.get("forecast_change_pct", 0)
    trend = payload.get("trend", "sideways")
    risk_factors = payload.get("risk_factors", [])
    news_headlines = payload.get("news_headlines", [])

    # Build summary from available data
    direction_word = "upward" if direction == "up" else "downward" if direction == "down" else "sideways"
    summary = (
        f"Technical indicators suggest a {trend} trend with a {direction_word} "
        f"forecast of {abs(change_pct):.1f}%. Algorithmic signal: {signal}."
    )

    trend_bullets = [
        f"Price forecast projects {direction_word} movement of {abs(change_pct):.1f}% over 14 days."
    ]
    if trend != "sideways":
        trend_bullets.append(f"Overall trend is {trend}, confirmed by moving average alignment.")

    # Simple divergence detection
    bearish_news = any(
        word in " ".join(news_headlines).lower()
        for word in ["decline", "fall", "drop", "risk", "warning", "concern", "lawsuit", "recall"]
    )
    bullish_signal = signal == "BUY"
    divergence = bearish_news and bullish_signal

    key_risks = risk_factors[:3] if risk_factors else ["Market volatility", "Limited historical data"]

    return {
        "summary": summary,
        "trend_explanation": trend_bullets,
        "divergence_warning": {
            "detected": divergence,
            "explanation": "News sentiment may conflict with bullish technical signal." if divergence else "",
        },
        "key_risks": key_risks,
        "disclaimer": (
            "This AI-generated synthesis is for informational purposes only "
            "and does not constitute financial advice. Consult a licensed advisor before investing."
        ),
    }


def get_llm_insight(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send compressed payload to OpenAI and return structured insight.
    Falls back to deterministic output if API is unavailable or fails.
    """
    if not _is_api_available():
        logger.info("OpenAI API key not configured — using deterministic fallback.")
        return _build_fallback_insight(payload)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        user_message = json.dumps(payload, separators=(",", ":"))  # compact JSON

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                )

                content = response.choices[0].message.content
                result = json.loads(content)

                # Validate required keys exist
                required = {"summary", "trend_explanation", "divergence_warning", "key_risks", "disclaimer"}
                if not required.issubset(result.keys()):
                    missing = required - result.keys()
                    logger.warning(f"LLM response missing keys: {missing}. Patching with defaults.")
                    fallback = _build_fallback_insight(payload)
                    for key in missing:
                        result[key] = fallback[key]

                return result

            except Exception as e:
                logger.warning(f"OpenAI API attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)

        # All retries exhausted
        logger.error("OpenAI API failed after retries — using fallback.")
        return _build_fallback_insight(payload)

    except ImportError:
        logger.error("openai package not installed — using fallback.")
        return _build_fallback_insight(payload)
    except Exception as e:
        logger.error(f"Unexpected LLM error: {e} — using fallback.")
        return _build_fallback_insight(payload)
