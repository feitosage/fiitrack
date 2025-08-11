
# Opcional: exemplo de coleta via BRAPI (https://brapi.dev)
# Preencha e adapte para seu uso; mantemos yfinance como default.
import os, requests, pandas as pd
BRAPI_TOKEN = os.getenv("BRAPI_TOKEN", "")

def fetch_brapi_prices(tickers):
    out = []
    for t in tickers:
        url = f"https://brapi.dev/api/quote/{t}?token={BRAPI_TOKEN}"
        r = requests.get(url, timeout=15)
        if r.ok:
            j = r.json()
            results = j.get("results", [])
            if results:
                out.append({
                    "ticker": t,
                    "close": results[0].get("regularMarketPrice"),
                    "dt": pd.Timestamp.utcnow().date()
                })
    return pd.DataFrame(out)
