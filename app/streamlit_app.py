
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text
import os
import sys

# Garante que a raiz do projeto esteja no sys.path para permitir imports como `from app...`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from app.db import get_engine, init_db
from app.ingest import update_all
from app.settings import DEFAULT_WATCHLIST
from app.data_sources import yf as yfd

st.set_page_config(page_title="FII Quote Tracker", layout="wide")
st.title("📈 FII Quote Tracker")

with st.sidebar:
    st.header("Configurações")
    tickers = st.text_input("FIIs (separados por vírgula):", value=",".join(DEFAULT_WATCHLIST)).upper().replace(" ", "").split(",")
    dias_lookback = st.selectbox(
        "Últimos dias",
        options=[5, 7, 15, 22, 30, 60, 90, 180, 252],
        index=4,  # 30 dias por padrão
    )
    if st.button("Atualizar dados agora"):
        with st.spinner("Baixando dados..."):
            n_prices, n_divs = update_all(tickers)
        st.success(f"Atualizado! {n_prices} preços e {n_divs} dividendos inseridos/atualizados.")

init_db()
engine = get_engine()

# Cartões: maior alta e maior baixa do dia (com base no último pregão disponível)
try:
    with engine.connect() as conn:
        df_rb = pd.read_sql(
            text(
                """
                SELECT ticker, dt, close
                FROM prices
                WHERE dt >= DATE((SELECT MAX(dt) FROM prices), '-7 day')
                ORDER BY ticker, dt
                """
            ),
            conn,
        )
    if not df_rb.empty:
        df_rb = df_rb[df_rb["ticker"].isin(tickers)].copy()
        if not df_rb.empty:
            df_rb["dt"] = pd.to_datetime(df_rb["dt"]).dt.date
            df_rb = df_rb.sort_values(["ticker", "dt"]).copy()
            df_rb["close_prev"] = df_rb.groupby("ticker")["close"].shift(1)
            last_dt = df_rb["dt"].max()
            today_df = df_rb[(df_rb["dt"] == last_dt) & df_rb["close_prev"].notna()].copy()
            if not today_df.empty:
                today_df["ret"] = (today_df["close"] / today_df["close_prev"]) - 1
                gainer = today_df.loc[today_df["ret"].idxmax()]
                loser = today_df.loc[today_df["ret"].idxmin()]
                cga, clo = st.columns(2)
                with cga:
                    st.metric(
                        f"Maior alta do dia — {str(gainer['ticker'])}",
                        f"R$ {gainer['close']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
                        f"{gainer['ret']*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."),
                    )
                with clo:
                    st.metric(
                        f"Maior baixa do dia — {str(loser['ticker'])}",
                        f"R$ {loser['close']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
                        f"{loser['ret']*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."),
                    )
except Exception as e:
    st.warning(f"Não foi possível calcular as maiores altas/baixas de hoje. {e}")

# Seleção principal
col1, col2 = st.columns([3,1])
with col2:
    sel = st.selectbox("Escolha o FII", options=tickers, index=0 if tickers else None)

if sel:
    with engine.connect() as conn:
        prices = pd.read_sql(text("SELECT * FROM prices WHERE ticker=:t ORDER BY dt"), conn, params={"t": sel})
        divs = pd.read_sql(text("SELECT * FROM dividends WHERE ticker=:t ORDER BY ex_date"), conn, params={"t": sel})
    if prices.empty:
        st.info("Sem dados ainda. Clique em **Atualizar dados agora**.")
    else:
        # Métricas rápidas
        last = prices.iloc[-1]["close"]
        wk = prices.tail(5)["close"].pct_change().dropna().sum()
        mo = prices.tail(22)["close"].pct_change().dropna().sum()
        y = prices.tail(252)["close"].pct_change().dropna().sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Preço atual", f"R$ {last:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
        m2.metric("Retorno 1M", f"{mo*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X","."))
        m3.metric("Retorno 1A", f"{y*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X","."))

        cutoff_sel = pd.to_datetime(prices["dt"]).max() - pd.Timedelta(days=int(dias_lookback))
        prices_plot = prices[pd.to_datetime(prices["dt"]) >= cutoff_sel].copy()

        fig = px.line(prices_plot, x="dt", y="close", title=f"Preço - {sel} (últimos {dias_lookback} dias)")
        st.plotly_chart(fig, use_container_width=True)

        # Gráfico de cotação diária (Candlestick OHLC)
        prices_ohlc = prices_plot.sort_values("dt").dropna(subset=["open","high","low","close"]).copy()
        if not prices_ohlc.empty:
            candle = go.Figure(
                data=[
                    go.Candlestick(
                        x=prices_ohlc["dt"],
                        open=prices_ohlc["open"],
                        high=prices_ohlc["high"],
                        low=prices_ohlc["low"],
                        close=prices_ohlc["close"],
                        increasing_line_color="#2CA02C",
                        decreasing_line_color="#D62728"
                    )
                ]
            )
            candle.update_layout(title=f"Cotação diária (OHLC) - {sel} (últimos {dias_lookback} dias)", xaxis_title="Data", yaxis_title="Preço")
            st.plotly_chart(candle, use_container_width=True)

        if not divs.empty:
            # DY trailing simples: soma dos últimos 12 meses / último preço
            divs["year_month"] = pd.to_datetime(divs["ex_date"]).dt.to_period("M")
            last12 = pd.to_datetime(prices["dt"]).max() - pd.offsets.DateOffset(years=1)
            dyt = divs[pd.to_datetime(divs["ex_date"]) >= last12]["value"].sum() / last
            st.metric("Dividend Yield (12m, aprox.)", f"{dyt*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X","."))
            bar = px.bar(divs, x="ex_date", y="value", title=f"Proventos - {sel}")
            st.plotly_chart(bar, use_container_width=True)

        # Intradiário de hoje (por hora)
        with st.expander("Cotação intradiária (hoje) por hora", expanded=True):
            intra = yfd.fetch_intraday_hourly(sel)
            if intra is None or intra.empty:
                st.info("Sem dados intradiários para hoje.")
            else:
                # Preparar dados limpos para o gráfico
                intra_plot = intra.copy()
                intra_plot = intra_plot.dropna(subset=["open","high","low","close"]).copy()
                x_vals = pd.to_datetime(intra_plot["dt"]).dt.tz_localize(None)
                o_vals = pd.to_numeric(intra_plot["open"], errors="coerce")
                h_vals = pd.to_numeric(intra_plot["high"], errors="coerce")
                l_vals = pd.to_numeric(intra_plot["low"], errors="coerce")
                c_vals = pd.to_numeric(intra_plot["close"], errors="coerce")

                intra_fig = go.Figure(data=[
                    go.Candlestick(
                        x=x_vals,
                        open=o_vals,
                        high=h_vals,
                        low=l_vals,
                        close=c_vals,
                        increasing_line_color="#2CA02C",
                        decreasing_line_color="#D62728"
                    )
                ])
                intra_fig.update_layout(title=f"Intradiário (hoje) - {sel} (60m)", xaxis_title="Hora", yaxis_title="Preço", xaxis_rangeslider_visible=False)
                st.plotly_chart(intra_fig, use_container_width=True)

                table = intra.copy()
                table["hora"] = pd.to_datetime(table["dt"]).dt.tz_localize(None).dt.strftime("%H:%M")
                st.dataframe(table[["hora","open","high","low","close","volume"]], use_container_width=True)

st.caption("Dados via Yahoo Finance (yfinance). DY e métricas são estimativas e não substituem análises oficiais.")
