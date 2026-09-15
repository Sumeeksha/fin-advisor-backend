"""
Technical Indicators Service
Computes RSI, MACD, Moving Averages, Bollinger Bands, Volume from OHLCV DataFrame.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(2)


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Dict[str, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {
        "macd": macd_line.round(4),
        "signal": signal_line.round(4),
        "histogram": histogram.round(4),
    }


def compute_bollinger(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Dict[str, pd.Series]:
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return {
        "upper": upper.round(2),
        "middle": sma.round(2),
        "lower": lower.round(2),
    }


def fetch_tradingview_indicators(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch official technical indicators directly from TradingView TA API."""
    try:
        from tradingview_ta import TA_Handler, Interval
        ticker = ticker.upper().strip()
        for ex in ["NASDAQ", "NYSE", "AMEX", ""]:
            try:
                handler = TA_Handler(
                    symbol=ticker,
                    exchange=ex,
                    screener="america",
                    interval=Interval.INTERVAL_1_DAY
                )
                analysis = handler.get_analysis()
                if analysis and analysis.indicators:
                    return analysis.indicators
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"tradingview-ta error for {ticker}: {e}")
    return None


def get_all_indicators(df: pd.DataFrame, ticker: str = "", source: str = "yfinance") -> Dict[str, Any]:
    """Compute all technical indicators and return structured dict."""
    if df is None or df.empty or len(df) < 30:
        return {"error": "Insufficient data to compute indicators"}

    close = df["Close"].dropna()
    volume = df["Volume"].dropna() if "Volume" in df.columns else pd.Series(dtype=float)

    # RSI
    rsi_series = compute_rsi(close)
    current_rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else None

    # MACD
    macd_data = compute_macd(close)
    current_macd = float(macd_data["macd"].iloc[-1])
    current_signal = float(macd_data["signal"].iloc[-1])
    current_histogram = float(macd_data["histogram"].iloc[-1])

    # Moving Averages
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    ema_12 = _ema(close, 12)
    ema_26 = _ema(close, 26)

    current_price = float(close.iloc[-1])

    def safe_last(series: pd.Series) -> Optional[float]:
        try:
            val = series.iloc[-1]
            return round(float(val), 2) if not np.isnan(val) else None
        except Exception:
            return None

    # Bollinger Bands
    bb = compute_bollinger(close)

    # Volume
    avg_vol_20 = float(volume.rolling(20).mean().iloc[-1]) if not volume.empty else None
    current_vol = float(volume.iloc[-1]) if not volume.empty else None

    # Trend determination
    trend = _determine_trend(current_price, safe_last(sma_50), safe_last(sma_200))

    # RSI signals for history (last 30 points)
    rsi_history = _series_to_list(rsi_series.tail(100))
    macd_history = _series_to_list(macd_data["macd"].tail(100))
    signal_history = _series_to_list(macd_data["signal"].tail(100))
    histogram_history = _series_to_list(macd_data["histogram"].tail(100))
    dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in close.tail(100).index]

    # Base result
    result = {
        "rsi": {
            "current": round(current_rsi, 2) if current_rsi else None,
            "signal": _rsi_signal(current_rsi),
            "history": rsi_history,
            "dates": dates,
        },
        "macd": {
            "macd": round(current_macd, 4),
            "signal": round(current_signal, 4),
            "histogram": round(current_histogram, 4),
            "crossover": "bullish" if current_macd > current_signal else "bearish",
            "macd_history": macd_history,
            "signal_history": signal_history,
            "histogram_history": histogram_history,
            "dates": dates,
        },
        "moving_averages": {
            "sma_20": safe_last(sma_20),
            "sma_50": safe_last(sma_50),
            "sma_200": safe_last(sma_200),
            "ema_12": safe_last(ema_12),
            "ema_26": safe_last(ema_26),
            "price_vs_sma50": round((current_price - safe_last(sma_50)) / safe_last(sma_50) * 100, 2) if safe_last(sma_50) else None,
            "price_vs_sma200": round((current_price - safe_last(sma_200)) / safe_last(sma_200) * 100, 2) if safe_last(sma_200) else None,
            "golden_cross": _golden_cross(sma_50, sma_200),
        },
        "bollinger_bands": {
            "upper": safe_last(bb["upper"]),
            "middle": safe_last(bb["middle"]),
            "lower": safe_last(bb["lower"]),
            "bandwidth": round((safe_last(bb["upper"]) - safe_last(bb["lower"])) / safe_last(bb["middle"]) * 100, 2)
            if safe_last(bb["middle"]) else None,
            "percent_b": round(
                (current_price - safe_last(bb["lower"])) / (safe_last(bb["upper"]) - safe_last(bb["lower"])) * 100, 2
            ) if safe_last(bb["upper"]) and safe_last(bb["lower"]) and safe_last(bb["upper"]) != safe_last(bb["lower"]) else None,
        },
        "volume": {
            "current": int(current_vol) if current_vol else None,
            "avg_20d": int(avg_vol_20) if avg_vol_20 else None,
            "relative": round(current_vol / avg_vol_20, 2) if current_vol and avg_vol_20 and avg_vol_20 > 0 else None,
        },
        "trend": trend,
        "current_price": current_price,
        "data_source": "yfinance",
    }

    # If TradingView is requested, fetch and merge official TradingView indicators
    if source.lower() == "tradingview" and ticker:
        tv_ind = fetch_tradingview_indicators(ticker)
        if tv_ind:
            result["data_source"] = "TradingView"
            if "RSI" in tv_ind and tv_ind["RSI"] is not None:
                tv_rsi = round(float(tv_ind["RSI"]), 2)
                result["rsi"]["current"] = tv_rsi
                result["rsi"]["signal"] = _rsi_signal(tv_rsi)

            if "MACD.macd" in tv_ind and tv_ind["MACD.macd"] is not None:
                tv_macd = round(float(tv_ind["MACD.macd"]), 4)
                tv_sig = round(float(tv_ind.get("MACD.signal", 0)), 4)
                result["macd"]["macd"] = tv_macd
                result["macd"]["signal"] = tv_sig
                result["macd"]["histogram"] = round(tv_macd - tv_sig, 4)
                result["macd"]["crossover"] = "bullish" if tv_macd > tv_sig else "bearish"

            ma_dict = result["moving_averages"]
            if "SMA20" in tv_ind and tv_ind["SMA20"]: ma_dict["sma_20"] = round(float(tv_ind["SMA20"]), 2)
            if "SMA50" in tv_ind and tv_ind["SMA50"]: ma_dict["sma_50"] = round(float(tv_ind["SMA50"]), 2)
            if "SMA200" in tv_ind and tv_ind["SMA200"]: ma_dict["sma_200"] = round(float(tv_ind["SMA200"]), 2)
            if "EMA12" in tv_ind and tv_ind["EMA12"]: ma_dict["ema_12"] = round(float(tv_ind["EMA12"]), 2)
            if "EMA26" in tv_ind and tv_ind["EMA26"]: ma_dict["ema_26"] = round(float(tv_ind["EMA26"]), 2)

            bb_dict = result["bollinger_bands"]
            if "BB.upper" in tv_ind and tv_ind["BB.upper"]: bb_dict["upper"] = round(float(tv_ind["BB.upper"]), 2)
            if "BB.lower" in tv_ind and tv_ind["BB.lower"]: bb_dict["lower"] = round(float(tv_ind["BB.lower"]), 2)

    return result


def _rsi_signal(rsi: Optional[float]) -> str:
    if rsi is None:
        return "neutral"
    if rsi < 30:
        return "oversold"
    if rsi > 70:
        return "overbought"
    if rsi < 45:
        return "bearish"
    if rsi > 55:
        return "bullish"
    return "neutral"


def _determine_trend(price: float, sma50: Optional[float], sma200: Optional[float]) -> str:
    if sma50 and sma200:
        if price > sma50 > sma200:
            return "bullish"
        if price < sma50 < sma200:
            return "bearish"
    if sma50:
        if price > sma50 * 1.02:
            return "bullish"
        if price < sma50 * 0.98:
            return "bearish"
    return "sideways"


def _golden_cross(sma50: pd.Series, sma200: pd.Series) -> Optional[str]:
    """Detect golden cross / death cross in recent 5 bars."""
    try:
        diff = (sma50 - sma200).dropna()
        if len(diff) < 6:
            return None
        recent = diff.iloc[-5:]
        if recent.iloc[-1] > 0 and recent.iloc[0] <= 0:
            return "golden_cross"
        if recent.iloc[-1] < 0 and recent.iloc[0] >= 0:
            return "death_cross"
        return "positive" if diff.iloc[-1] > 0 else "negative"
    except Exception:
        return None


def _series_to_list(series: pd.Series) -> List[Optional[float]]:
    return [round(float(v), 4) if not np.isnan(v) else None for v in series]
