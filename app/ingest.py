
from sqlalchemy import text
from app.db import get_engine, init_db
from app.data_sources import yf as yfd
from typing import List

def upsert_prices(df):
    if df is None or df.empty: return 0
    engine = get_engine()
    rows = 0
    with engine.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(text('''
                INSERT INTO prices (ticker, dt, close, open, high, low, volume)
                VALUES (:ticker, :dt, :close, :open, :high, :low, :volume)
                ON CONFLICT(ticker, dt) DO UPDATE SET
                    close=excluded.close,
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    volume=excluded.volume;
            '''), r.to_dict())
            rows += 1
    return rows

def upsert_dividends(df):
    if df is None or df.empty: return 0
    engine = get_engine()
    rows = 0
    with engine.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(text('''
                INSERT INTO dividends (ticker, ex_date, value)
                VALUES (:ticker, :ex_date, :value)
                ON CONFLICT(ticker, ex_date) DO UPDATE SET
                    value=excluded.value;
            '''), r.to_dict())
            rows += 1
    return rows

def update_all(tickers: List[str]):
    init_db()
    hist = yfd.fetch_history(tickers, period="2y", interval="1d")
    divs = yfd.fetch_dividends(tickers)
    n_prices = upsert_prices(hist)
    n_divs = upsert_dividends(divs)
    return n_prices, n_divs
