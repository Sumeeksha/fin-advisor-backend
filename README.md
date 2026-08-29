# FinAdvisor 🚀 — Full-Stack Stock Analysis & AI Investment Advisor

FinAdvisor is a modern full-stack web application that provides real-time stock analysis, historical trend evaluation, interactive visualizations, and decision-support investment advice. It translates complex financial data into plain-language actionable insights.

---

## 🛠️ Architecture & Tech Stack

- **Backend**: Python (FastAPI), uvicorn, yfinance, pandas, numpy, statsmodels (ARIMA)
- **Frontend**: Next.js 16 (App Router, Tailwind CSS, Turbopack, Chart.js)
- **Dynamic Fallbacks**: A seamless **Sandbox Mode** triggers automatically if live APIs are rate-limited or missing keys. It models stock prices using geometric brownian random walks so that charts, ARIMA forecasts, indicators, and advice panels render flawlessly.

---

## 🔑 API Keys Registration Guide

FinAdvisor works completely out of the box with Yahoo Finance (via `yfinance` - no key needed). To enhance live quote quality and technical calculations, follow these steps to register for free keys:

### 1. Finnhub API Key (For Live Quotes & News)
1. Go to [finnhub.io/register](https://finnhub.io/register)
2. Enter your email and sign up.
3. Copy your API Key from the dashboard.
4. Add it to `backend/.env`:
   ```env
   FINNHUB_API_KEY=YOUR_KEY
   ```

### 2. Alpha Vantage API Key (For Technical Indicators)
1. Go to [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Choose the **Free API Key** tier.
3. Copy the generated key and add it to `backend/.env`:
   ```env
   ALPHA_VANTAGE_API_KEY=YOUR_KEY
   ```

---

## 🏃 Setup & Run

The application is already running in your background! You can open it here:
- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Command to restart both servers:
```bash
./start.sh
```
Use `Ctrl + C` in the terminal to stop both servers safely.

---

## 📊 Features

1. **Smart Autocomplete Search**: Debounced search for US equity symbols.
2. **Interactive Stock Charts**: Line/Area price history and volume bars with customizable timeframes (`1D`, `1W`, `1M`, `3M`, `1Y`, `5Y`).
3. **AI Advisor Panel**: Buy/Hold/Sell signals, entry/exit target levels, and structured evidence reasoning.
4. **Technical Indicator Metering**: RSI gauge, MACD crossover histograms, Bollinger Bands percent placement, and relative volume bars.
5. **ARIMA Price Forecasting**: 14-day price projections with 80% confidence interval shading bands.
