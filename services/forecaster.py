"""
Forecasting Service — Short-term price prediction using Linear Regression + ARIMA.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

DISCLAIMER = (
    "⚠️ Price forecasts are statistical projections based on historical data. "
    "They are probabilistic in nature and NOT guaranteed financial advice. "
    "Markets are inherently unpredictable. Consult a licensed financial advisor."
)


def generate_forecast(df: pd.DataFrame, days: int = 14, latest_price: float = None) -> Dict[str, Any]:
    """Generate 14-day price forecast using Linear Regression + optional ARIMA.
    If latest_price is provided (e.g. from a live TradingView quote), it is
    appended to the close series so the model incorporates the most recent tick.
    """
    if df is None or df.empty or len(df) < 30:
        return {"error": "Insufficient historical data for forecasting", "disclaimer": DISCLAIMER}

    close = df["Close"].dropna()

    # ── Inject latest live price into the series ──
    if latest_price is not None:
        now_ts = pd.Timestamp.now()
        # Only append if the latest price differs from the last entry
        if abs(close.iloc[-1] - latest_price) > 0.005:
            live_point = pd.Series([latest_price], index=[now_ts], name="Close")
            close = pd.concat([close, live_point])

    dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in close.index]

    # ── Linear Regression Forecast ────────────
    lr_forecast = _linear_regression_forecast(close, days)

    # ── ARIMA Forecast (best-effort) ──────────
    arima_forecast = _arima_forecast(close, days)

    # ── Historical data for chart ─────────────
    history_dates = dates[-90:]  # last 90 days
    history_prices = [round(float(p), 2) for p in close.tail(90)]

    # ── Combined forecast ─────────────────────
    # Use ARIMA if available, else linear regression
    primary = arima_forecast if arima_forecast else lr_forecast

    current_p = float(close.iloc[-1])
    proj_p = primary["prices"][-1] if primary.get("prices") else current_p
    upper_95 = primary["upper"][-1] if primary.get("upper") else round(proj_p * 1.03, 2)
    lower_95 = primary["lower"][-1] if primary.get("lower") else round(proj_p * 0.97, 2)

    # ── Derive ARIMA statistics dynamically ───
    aic_val = primary.get("aic", None)
    rmse_val = primary.get("rmse", None)
    drift_val = primary.get("drift_term", None)
    p_val = primary.get("p_value", None)
    order_str = primary.get("order", "(1,1,0)")
    econ_spec = primary.get("econometric_specification", None)
    method_str = f"ARIMA {order_str}" if arima_forecast else "Linear Regression"

    return {
        "method": method_str,
        "order": order_str,
        "econometric_specification": econ_spec,
        "aic_score": aic_val,
        "rmse": rmse_val,
        "drift_term": drift_val,
        "p_value": p_val,
        "confidence_corridor": {
            "lower_95": lower_95,
            "spot": current_p,
            "upper_95": upper_95,
        },
        "history": {
            "dates": history_dates,
            "prices": history_prices,
        },
        "forecast": {
            "dates": primary["dates"],
            "prices": primary["prices"],
            "upper_band": primary.get("upper", []),
            "lower_band": primary.get("lower", []),
        },
        "linear_regression": lr_forecast,
        "summary": _forecast_summary(close, primary["prices"]),
        "disclaimer": DISCLAIMER,
    }


def _linear_regression_forecast(close: pd.Series, days: int) -> Dict[str, Any]:
    """Simple linear regression on last 90 days."""
    try:
        data = close.tail(90).values
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        slope, intercept = coeffs[0], coeffs[1]

        # Project forward
        future_x = np.arange(len(data), len(data) + days)
        predicted = slope * future_x + intercept

        # Residuals for confidence bands
        fitted = slope * x + intercept
        residuals = data - fitted
        std_residual = np.std(residuals)

        forecast_dates = _future_dates(close.index[-1], days)
        prices = [round(float(p), 2) for p in predicted]
        upper = [round(float(p + 1.96 * std_residual), 2) for p in predicted]
        lower = [round(float(p - 1.96 * std_residual), 2) for p in predicted]

        return {
            "dates": forecast_dates,
            "prices": prices,
            "upper": upper,
            "lower": lower,
            "r_squared": _r_squared(data, fitted),
            "direction": "up" if slope > 0 else "down",
            "slope_per_day": round(float(slope), 4),
        }
    except Exception as e:
        logger.warning(f"Linear regression failed: {e}")
        return {}


def _arima_forecast(close: pd.Series, days: int) -> Optional[Dict[str, Any]]:
    """ARIMA(2,1,2) forecast — best-effort, fails gracefully.
    Returns computed model statistics (AIC, RMSE, drift, p-value, econometric spec)."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        data = close.tail(120).values
        order = (2, 1, 2)
        model = ARIMA(data, order=order)
        result = model.fit()
        forecast_obj = result.get_forecast(steps=days)
        mean_forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=0.2)  # 80% CI

        forecast_dates = _future_dates(close.index[-1], days)

        # ── Compute actual statistics from fitted model ──
        # RMSE from residuals
        residuals = result.resid
        rmse = round(float(np.sqrt(np.mean(residuals ** 2))), 2)

        # Drift term (mean daily change from the forecast)
        if len(mean_forecast) >= 2:
            avg_daily_change = (mean_forecast[-1] - float(data[-1])) / days
            drift_str = f"{'+' if avg_daily_change >= 0 else ''}{avg_daily_change:.2f}$/day"
        else:
            drift_str = "+0.00$/day"

        # ADF test for stationarity p-value on differenced series
        try:
            diff_data = np.diff(data)
            adf_result = adfuller(diff_data, maxlag=5)
            adf_pval = adf_result[1]
            p_val_str = f"{adf_pval:.4f} ({'Stationary' if adf_pval < 0.05 else 'Non-Stationary'})"
        except Exception:
            p_val_str = "N/A"

        # Econometric specification from fitted parameters
        params = result.params
        order_str = f"({order[0]},{order[1]},{order[2]})"
        try:
            spec_parts = []
            # Intercept / const
            if "const" in result.param_names:
                const_idx = list(result.param_names).index("const")
                spec_parts.append(f"{params[const_idx]:.2f}")
            # AR terms
            for i in range(order[0]):
                key = f"ar.L{i+1}"
                if key in result.param_names:
                    idx = list(result.param_names).index(key)
                    coef = params[idx]
                    sign = "+" if coef >= 0 else "-"
                    spec_parts.append(f"{sign} {abs(coef):.2f}ΔY_{{t-{i+1}}}")
            # MA terms
            for i in range(order[2]):
                key = f"ma.L{i+1}"
                if key in result.param_names:
                    idx = list(result.param_names).index(key)
                    coef = params[idx]
                    sign = "+" if coef >= 0 else "-"
                    spec_parts.append(f"{sign} {abs(coef):.2f}ε_{{t-{i+1}}}")
            econ_spec = "ΔY_t = " + " ".join(spec_parts) if spec_parts else None
        except Exception:
            econ_spec = None

        return {
            "dates": forecast_dates,
            "prices": [round(float(p), 2) for p in mean_forecast],
            "upper": [round(float(p), 2) for p in conf_int[:, 1]],
            "lower": [round(float(p), 2) for p in conf_int[:, 0]],
            "aic": round(float(result.aic), 2),
            "rmse": rmse,
            "drift_term": drift_str,
            "p_value": p_val_str,
            "order": order_str,
            "econometric_specification": econ_spec,
            "direction": "up" if mean_forecast[-1] > float(close.iloc[-1]) else "down",
        }
    except Exception as e:
        logger.info(f"ARIMA forecast unavailable (using LR instead): {e}")
        return None


def _future_dates(last_date, days: int) -> List[str]:
    """Generate future business day dates."""
    dates = []
    current = pd.Timestamp(last_date) + timedelta(days=1)
    added = 0
    while added < days:
        if current.weekday() < 5:  # Monday=0, Friday=4
            dates.append(str(current.date()))
            added += 1
        current += timedelta(days=1)
    return dates


def _r_squared(actual: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = np.sum((actual - fitted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 0.0
    return round(float(1 - ss_res / ss_tot), 4)


def _forecast_summary(close: pd.Series, forecast_prices: List[float]) -> Dict[str, Any]:
    current = float(close.iloc[-1])
    if not forecast_prices:
        return {}
    end_price = forecast_prices[-1]
    change_pct = (end_price - current) / current * 100
    return {
        "current_price": round(current, 2),
        "projected_price": round(end_price, 2),
        "projected_change_pct": round(change_pct, 2),
        "direction": "up" if change_pct > 0 else "down",
        "outlook": "Bullish" if change_pct > 3 else "Bearish" if change_pct < -3 else "Neutral",
    }
