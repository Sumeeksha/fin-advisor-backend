"""
LLM Client — Multi-provider API router for financial narrative synthesis.
Supports: OpenAI (GPT-4o), Google Gemini, Anthropic Claude, DeepSeek, and HuggingFace (FinGPT/FinMA).
Token-optimized with temperature=0.1, max_tokens=350, and JSON output formatting.
Provides deterministic fallback when API keys are absent or network requests fail.
"""

import os
import json
import logging
import time
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── API Key Configuration ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

TEMPERATURE = 0.1
MAX_TOKENS = 350
MAX_RETRIES = 1
RETRY_DELAY_S = 1.5

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


def _build_fallback_insight(payload: Dict[str, Any], provider_name: str = "Deterministic Fallback") -> Dict[str, Any]:
    """Deterministic fallback when LLM API keys are unconfigured or unavailable."""
    signal = payload.get("signal", "HOLD")
    direction = payload.get("forecast_direction", "flat")
    change_pct = payload.get("forecast_change_pct", 0) or 0.0
    trend = payload.get("trend", "sideways")
    risk_factors = payload.get("risk_factors", [])
    news_headlines = payload.get("news_headlines", [])

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
        "provider": provider_name,
        "is_fallback": True,
    }


def _validate_and_clean_json(raw_content: str) -> Optional[Dict[str, Any]]:
    """Sanitize LLM output and parse structured JSON dictionary."""
    if not raw_content:
        return None
    
    clean_str = raw_content.strip()
    if clean_str.startswith("```json"):
        clean_str = clean_str[7:]
    if clean_str.startswith("```"):
        clean_str = clean_str[3:]
    if clean_str.endswith("```"):
        clean_str = clean_str[:-3]
    clean_str = clean_str.strip()

    try:
        data = json.loads(clean_str)
        required = {"summary", "trend_explanation", "divergence_warning", "key_risks", "disclaimer"}
        if required.issubset(data.keys()):
            return data
    except Exception as e:
        logger.warning(f"Failed to parse LLM JSON output: {e}")
    return None


# ── Provider Specific Handlers ─────────────────────────────────────────

def _call_openai(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Execute real call to OpenAI GPT-4o API."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_KEY_HERE":
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        user_message = json.dumps(payload, separators=(",", ":"))

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        parsed = _validate_and_clean_json(response.choices[0].message.content)
        if parsed:
            parsed["provider"] = "OpenAI GPT-4o"
            parsed["is_fallback"] = False
        return parsed
    except Exception as e:
        logger.warning(f"OpenAI API call failed: {e}")
        return None


def _call_gemini(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Execute real call to Google Gemini 1.5 Pro REST API."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_KEY_HERE":
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"
        user_prompt = f"{SYSTEM_PROMPT}\n\nStock Payload Data:\n{json.dumps(payload)}"
        
        body = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": MAX_TOKENS,
                "responseMimeType": "application/json"
            }
        }
        res = requests.post(url, json=body, timeout=10)
        if res.status_code == 200:
            content = res.json()["candidates"][0]["content"]["parts"][0]["text"]
            parsed = _validate_and_clean_json(content)
            if parsed:
                parsed["provider"] = "Google Gemini 1.5 Pro"
                parsed["is_fallback"] = False
            return parsed
        else:
            logger.warning(f"Gemini API returned status {res.status_code}: {res.text}")
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")
    return None


def _call_claude(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Execute real call to Anthropic Claude 3.5 Sonnet Messages API."""
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "YOUR_ANTHROPIC_KEY_HERE":
        return None
    try:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": json.dumps(payload)}]
        }
        res = requests.post(url, headers=headers, json=body, timeout=10)
        if res.status_code == 200:
            content = res.json()["content"][0]["text"]
            parsed = _validate_and_clean_json(content)
            if parsed:
                parsed["provider"] = "Anthropic Claude 3.5 Sonnet"
                parsed["is_fallback"] = False
            return parsed
        else:
            logger.warning(f"Anthropic API status {res.status_code}: {res.text}")
    except Exception as e:
        logger.warning(f"Anthropic API call failed: {e}")
    return None


def _call_deepseek(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Execute real call to DeepSeek Chat API."""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_KEY_HERE":
        return None
    try:
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)}
            ],
            "response_format": {"type": "json_object"},
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS
        }
        res = requests.post(url, headers=headers, json=body, timeout=10)
        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"]
            parsed = _validate_and_clean_json(content)
            if parsed:
                parsed["provider"] = "DeepSeek V3"
                parsed["is_fallback"] = False
            return parsed
        else:
            logger.warning(f"DeepSeek API status {res.status_code}: {res.text}")
    except Exception as e:
        logger.warning(f"DeepSeek API call failed: {e}")
    return None


def _call_fingpt(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Execute real call to FinGPT via Hugging Face Inference API."""
    if not HF_TOKEN or HF_TOKEN == "YOUR_HF_TOKEN_HERE":
        return None
    try:
        url = "https://api-inference.huggingface.co/models/AI4Finance-Foundation/FinGPT-v3"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        prompt = f"{SYSTEM_PROMPT}\nInput: {json.dumps(payload)}"
        
        res = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=10)
        if res.status_code == 200:
            result = res.json()
            raw_text = result[0]["generated_text"] if isinstance(result, list) and result else str(result)
            parsed = _validate_and_clean_json(raw_text)
            if parsed:
                parsed["provider"] = "FinGPT (AI4Finance)"
                parsed["is_fallback"] = False
            return parsed
    except Exception as e:
        logger.warning(f"FinGPT HuggingFace call failed: {e}")
    return None


# ── Unified Public Interface ──────────────────────────────────────────

def get_llm_insight(payload: Dict[str, Any], provider: str = "dual") -> Dict[str, Any]:
    """
    Generate LLM-synthesized narrative insight.
    Routes to requested provider ('gpt4o', 'gemini', 'claude', 'deepseek', 'fingpt', 'dual').
    Falls back gracefully if key is unconfigured or request fails.
    """
    provider_clean = (provider or "dual").lower().strip()

    # Route request based on selected provider
    if provider_clean in ["gpt4o", "openai"]:
        res = _call_openai(payload)
        if res: return res

    elif provider_clean == "gemini":
        res = _call_gemini(payload)
        if res: return res

    elif provider_clean in ["claude", "anthropic"]:
        res = _call_claude(payload)
        if res: return res

    elif provider_clean == "deepseek":
        res = _call_deepseek(payload)
        if res: return res

    elif provider_clean in ["fingpt", "finma", "alli"]:
        res = _call_fingpt(payload)
        if res: return res

    # 'dual' or fallback sequence across available providers
    for caller in [_call_openai, _call_gemini, _call_claude, _call_deepseek, _call_fingpt]:
        res = caller(payload)
        if res:
            return res

    # If all configured APIs are absent or failed, return clean deterministic fallback
    logger.info(f"No active API keys found for '{provider_clean}' — using deterministic math fallback.")
    return _build_fallback_insight(payload, provider_name=f"Deterministic Fallback ({provider_clean.upper()})")
