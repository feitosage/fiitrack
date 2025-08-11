
import pandas as pd
import yfinance as yf
from typing import List, Tuple

def normalize_tickers(tickers: List[str]) -> List[str]:
    # Yahoo Finance usa sufixo .SA para B3
    return [t.upper().strip() + ".SA" if not t.upper().strip().endswith(".SA") else t.upper().strip() for t in tickers]

def fetch_history(tickers: List[str], period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    y_tickers = normalize_tickers(tickers)
    data = yf.download(y_tickers, period=period, interval=interval, group_by="ticker", auto_adjust=False, threads=True)
    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        for t in y_tickers:
            df = data[t].copy()
            df["ticker"] = t.replace(".SA", "")
            df = df.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
            df = df.reset_index().rename(columns={"Date": "dt"})
            frames.append(df[["ticker","dt","close","open","high","low","volume"]])
    else:
        # single ticker
        df = data.copy()
        df["ticker"] = y_tickers[0].replace(".SA", "")
        df = df.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
        df = df.reset_index().rename(columns={"Date": "dt"})
        frames.append(df[["ticker","dt","close","open","high","low","volume"]])
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out["dt"] = pd.to_datetime(out["dt"]).dt.date
        return out.sort_values(["ticker","dt"]).reset_index(drop=True)
    return pd.DataFrame(columns=["ticker","dt","close","open","high","low","volume"])

def fetch_dividends(tickers: List[str]) -> pd.DataFrame:
    records = []
    for t in normalize_tickers(tickers):
        tk = yf.Ticker(t)
        divs = tk.dividends  # pandas Series indexed by date
        if divs is None or len(divs) == 0:
            continue
        df = divs.reset_index()
        df.columns = ["ex_date", "value"]
        df["ticker"] = t.replace(".SA", "")
        df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.date
        records.append(df[["ticker","ex_date","value"]])
    if records:
        return pd.concat(records, ignore_index=True).sort_values(["ticker","ex_date"]).reset_index(drop=True)
    return pd.DataFrame(columns=["ticker","ex_date","value"])

def fetch_intraday_hourly(ticker: str) -> pd.DataFrame:
    """Retorna candles horários do dia atual (ou último dia disponível) para um ticker (B3) via Yahoo Finance.

    Estratégia: tenta intervalos 60m, 30m e 15m; se menor que 60m, reamostra para 1h.
    Colunas: dt (datetime tz America/Sao_Paulo), open, high, low, close, volume
    """
    y_ticker = normalize_tickers([ticker])[0]

    data = None
    chosen_interval = None
    for interval in ["60m", "30m", "15m"]:
        try:
            tmp = yf.download(
                y_ticker,
                period="5d",
                interval=interval,
                auto_adjust=False,
                prepost=False,
                threads=True,
                progress=False,
            )
        except Exception:
            tmp = None
        if tmp is not None and not tmp.empty:
            data = tmp
            chosen_interval = interval
            break

    if data is None or data.empty:
        return pd.DataFrame(columns=["dt", "open", "high", "low", "close", "volume"])

    # Extrair times do índice e normalizar timezone
    time_index = data.index
    if getattr(time_index, "tz", None) is not None:
        dt_utc = time_index.tz_convert("UTC")
    else:
        dt_utc = pd.to_datetime(time_index, utc=True, errors="coerce")
    dt_sp = dt_utc.tz_convert("America/Sao_Paulo")

    # Selecionar colunas OHLCV, lidando com possíveis MultiIndex variados
    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        # Descobrir qual nível tem os campos e qual tem os tickers
        field_names = {"Open", "High", "Low", "Close", "Volume"}
        level_with_fields = None
        for lvl in range(cols.nlevels):
            values = set(map(str, cols.get_level_values(lvl)))
            if field_names.issubset(values):
                level_with_fields = lvl
                break
        if level_with_fields is None:
            # Falha incomum: achatar colunas e tentar renomear depois
            data_single = data.copy()
            data_single.columns = ["_".join(map(str, c)) for c in cols]
            # Procurar diretamente por nomes contendo cada campo
            mapping = {}
            for f in field_names:
                for c in data_single.columns:
                    if c.endswith(f):
                        mapping[f] = c
                        break
            ohlcv = data_single[[mapping.get("Open"), mapping.get("High"), mapping.get("Low"), mapping.get("Close"), mapping.get("Volume")]].copy()
            ohlcv.columns = ["Open", "High", "Low", "Close", "Volume"]
        else:
            # Nível de tickers é o outro nível (assumindo 2 níveis)
            level_with_tickers = 1 - level_with_fields if cols.nlevels == 2 else (0 if level_with_fields != 0 else 1)
            mask_ticker = cols.get_level_values(level_with_tickers) == y_ticker
            # Se não encontrar exatamente o ticker, tenta qualquer um existente (único)
            if not mask_ticker.any():
                unique_tickers = list(dict.fromkeys(cols.get_level_values(level_with_tickers)))
                if unique_tickers:
                    mask_ticker = cols.get_level_values(level_with_tickers) == unique_tickers[0]
            data_single = data.loc[:, mask_ticker]
            # Reduzir para campos no nível de fields
            if isinstance(data_single.columns, pd.MultiIndex):
                data_single.columns = data_single.columns.get_level_values(level_with_fields)
            ohlcv = data_single.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    else:
        ohlcv = data.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()

    df = ohlcv.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df = df.assign(dt=dt_sp)

    # Filtrar dia alvo
    today_sp = pd.Timestamp.now(tz="America/Sao_Paulo").date()
    dates = df["dt"].dt.date
    target_date = today_sp if (dates == today_sp).any() else dates.max()
    df_day = df.loc[dates == target_date, ["dt", "open", "high", "low", "close", "volume"]].copy()

    if df_day.empty:
        return df_day

    # Se intervalo não for 60m, reamostrar para 1H consolidando OHLC e soma de volume
    if chosen_interval in {"30m", "15m"}:
        df_day = (
            df_day.set_index("dt")
            .resample("1H", label="right", closed="right")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna(subset=["open", "high", "low", "close"])  # remover janelas vazias
            .reset_index()
        )

    # Garantir numéricos e remover quaisquer linhas com NaN em OHLC
    for col in ["open", "high", "low", "close", "volume"]:
        df_day[col] = pd.to_numeric(df_day[col], errors="coerce")
    df_day = df_day.dropna(subset=["open", "high", "low", "close"]).sort_values("dt")

    return df_day.reset_index(drop=True)
