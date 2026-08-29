"""
Investment Advisor Service
Generates Buy / Hold / Sell signals with evidence-based reasoning.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

SIGNAL_BUY = "BUY"
SIGNAL_HOLD = "HOLD"
SIGNAL_SELL = "SELL"


def generate_advice(quote: Dict[str, Any], indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a structured investment advice object.
    Returns signal, confidence, reasoning bullets, and entry/exit ranges.
    """
    if "error" in indicators:
        return _neutral_advice(quote, "Insufficient historical data for analysis")

    score = 0  # positive = bullish, negative = bearish
    max_score = 0
    reasons: List[Dict[str, str]] = []
    risk_factors: List[str] = []

    price = quote.get("price", 0)
    change_pct = quote.get("change_pct", 0)

    # ── RSI Analysis ──────────────────────────
    rsi = indicators.get("rsi", {})
    rsi_val = rsi.get("current")
    if rsi_val is not None:
        max_score += 2
        if rsi_val < 30:
            score += 2
            reasons.append({
                "icon": "📉",
                "text": f"RSI={rsi_val:.1f} — Oversold territory (< 30). Strong mean-reversion buy signal.",
                "sentiment": "bullish",
            })
        elif rsi_val < 40:
            score += 1
            reasons.append({
                "icon": "📊",
                "text": f"RSI={rsi_val:.1f} — Approaching oversold. Potential accumulation zone.",
                "sentiment": "bullish",
            })
        elif rsi_val > 70:
            score -= 2
            reasons.append({
                "icon": "📈",
                "text": f"RSI={rsi_val:.1f} — Overbought (> 70). Risk of pullback or correction.",
                "sentiment": "bearish",
            })
            risk_factors.append("Overbought RSI")
        elif rsi_val > 60:
            score -= 1
            reasons.append({
                "icon": "📊",
                "text": f"RSI={rsi_val:.1f} — Approaching overbought. Monitor for reversal signals.",
                "sentiment": "neutral",
            })
        else:
            reasons.append({
                "icon": "📊",
                "text": f"RSI={rsi_val:.1f} — Neutral momentum zone (30–60).",
                "sentiment": "neutral",
            })

    # ── MACD Analysis ──────────────────────────
    macd = indicators.get("macd", {})
    macd_val = macd.get("macd")
    signal_val = macd.get("signal")
    histogram = macd.get("histogram")
    if macd_val is not None and signal_val is not None:
        max_score += 2
        crossover = macd.get("crossover")
        if crossover == "bullish":
            score += 2
            reasons.append({
                "icon": "✅",
                "text": f"MACD bullish crossover — MACD ({macd_val:.4f}) above Signal ({signal_val:.4f}). Upward momentum building.",
                "sentiment": "bullish",
            })
        else:
            score -= 2
            reasons.append({
                "icon": "⚠️",
                "text": f"MACD bearish crossover — MACD ({macd_val:.4f}) below Signal ({signal_val:.4f}). Downward pressure.",
                "sentiment": "bearish",
            })
            risk_factors.append("MACD bearish crossover")

    # ── Moving Average Analysis ─────────────────
    ma = indicators.get("moving_averages", {})
    sma50 = ma.get("sma_50")
    sma200 = ma.get("sma_200")
    golden = ma.get("golden_cross")
    max_score += 2

    if sma50 and price > sma50:
        score += 1
        reasons.append({
            "icon": "📐",
            "text": f"Price (${price:.2f}) is above the 50-day SMA (${sma50:.2f}) — short-term uptrend intact.",
            "sentiment": "bullish",
        })
    elif sma50:
        score -= 1
        reasons.append({
            "icon": "📐",
            "text": f"Price (${price:.2f}) is below the 50-day SMA (${sma50:.2f}) — short-term downtrend.",
            "sentiment": "bearish",
        })
        risk_factors.append("Price below 50-day SMA")

    if golden == "golden_cross":
        score += 1
        reasons.append({
            "icon": "⭐",
            "text": "Golden Cross detected — 50-day SMA crossed above 200-day SMA. Long-term bullish signal.",
            "sentiment": "bullish",
        })
    elif golden == "death_cross":
        score -= 1
        reasons.append({
            "icon": "💀",
            "text": "Death Cross detected — 50-day SMA crossed below 200-day SMA. Long-term bearish warning.",
            "sentiment": "bearish",
        })
        risk_factors.append("Death Cross pattern")

    # ── Bollinger Bands ─────────────────────────
    bb = indicators.get("bollinger_bands", {})
    pct_b = bb.get("percent_b")
    max_score += 1
    if pct_b is not None:
        if pct_b < 10:
            score += 1
            reasons.append({
                "icon": "🔵",
                "text": f"Price near lower Bollinger Band (%B={pct_b:.1f}%) — potential support bounce.",
                "sentiment": "bullish",
            })
        elif pct_b > 90:
            score -= 1
            reasons.append({
                "icon": "🔴",
                "text": f"Price near upper Bollinger Band (%B={pct_b:.1f}%) — potential resistance rejection.",
                "sentiment": "bearish",
            })
            risk_factors.append("Near upper Bollinger Band")

    # ── Volume Confirmation ─────────────────────
    vol = indicators.get("volume", {})
    rel_vol = vol.get("relative")
    max_score += 1
    if rel_vol is not None:
        if rel_vol > 1.5 and change_pct > 0:
            score += 1
            reasons.append({
                "icon": "📦",
                "text": f"High volume breakout ({rel_vol:.1f}x average) confirming price move — institutional interest.",
                "sentiment": "bullish",
            })
        elif rel_vol > 1.5 and change_pct < 0:
            score -= 1
            reasons.append({
                "icon": "📦",
                "text": f"High volume sell-off ({rel_vol:.1f}x average) — distribution pattern detected.",
                "sentiment": "bearish",
            })
            risk_factors.append("High-volume sell-off")
        else:
            reasons.append({
                "icon": "📦",
                "text": f"Volume at {rel_vol:.1f}x 20-day average — {'confirming' if change_pct > 0 else 'normal'} trading conditions.",
                "sentiment": "neutral",
            })

    # ── Trend ────────────────────────────────────
    trend = indicators.get("trend", "sideways")

    # ── Signal Determination ─────────────────────
    confidence_raw = abs(score) / max_score if max_score > 0 else 0
    confidence = min(int(confidence_raw * 100), 95)

    if score >= 3:
        signal = SIGNAL_BUY
    elif score <= -3:
        signal = SIGNAL_SELL
    else:
        signal = SIGNAL_HOLD

    # ── Entry / Exit Ranges ──────────────────────
    entry_low, entry_high, exit_target, stop_loss = _compute_levels(price, indicators, signal)

    return {
        "signal": signal,
        "confidence": confidence,
        "score": score,
        "max_score": max_score,
        "trend": trend,
        "reasons": reasons,
        "risk_factors": risk_factors,
        "entry_range": {"low": entry_low, "high": entry_high},
        "exit_target": exit_target,
        "stop_loss": stop_loss,
        "disclaimer": (
            "This analysis is based on technical indicators only and is for informational purposes. "
            "It does not constitute financial advice. Past performance does not guarantee future results. "
            "Always consult a licensed financial advisor before investing."
        ),
    }


def _compute_levels(
    price: float,
    indicators: Dict[str, Any],
    signal: str,
) -> tuple:
    """Compute entry range, exit target, and stop loss based on support/resistance."""
    bb = indicators.get("bollinger_bands", {})
    bb_lower = bb.get("lower") or price * 0.97
    bb_upper = bb.get("upper") or price * 1.05
    bb_middle = bb.get("middle") or price

    ma = indicators.get("moving_averages", {})
    sma50 = ma.get("sma_50") or price
    sma200 = ma.get("sma_200") or price

    if signal == SIGNAL_BUY:
        entry_low = round(max(bb_lower, price * 0.97), 2)
        entry_high = round(min(price * 1.01, sma50 * 1.005), 2)
        exit_target = round(min(bb_upper, price * 1.08), 2)
        stop_loss = round(entry_low * 0.97, 2)
    elif signal == SIGNAL_SELL:
        entry_low = round(price * 0.99, 2)
        entry_high = round(price * 1.01, 2)
        exit_target = round(max(bb_lower, price * 0.94), 2)
        stop_loss = round(price * 1.04, 2)
    else:  # HOLD
        entry_low = round(price * 0.97, 2)
        entry_high = round(price * 1.01, 2)
        exit_target = round(price * 1.06, 2)
        stop_loss = round(price * 0.95, 2)

    return entry_low, entry_high, exit_target, stop_loss


def _neutral_advice(quote: Dict[str, Any], reason: str) -> Dict[str, Any]:
    price = quote.get("price", 0)
    return {
        "signal": SIGNAL_HOLD,
        "confidence": 0,
        "score": 0,
        "max_score": 0,
        "trend": "unknown",
        "reasons": [{"icon": "ℹ️", "text": reason, "sentiment": "neutral"}],
        "risk_factors": [],
        "entry_range": {"low": round(price * 0.97, 2), "high": round(price * 1.01, 2)},
        "exit_target": round(price * 1.06, 2),
        "stop_loss": round(price * 0.95, 2),
        "disclaimer": "Insufficient data for full analysis. Consult a licensed financial advisor.",
    }
