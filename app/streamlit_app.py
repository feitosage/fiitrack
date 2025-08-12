
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
        options=["Todo histórico", 5, 7, 15, 22, 30, 60, 90, 180, 252],
        index=5,  # 30 dias por padrão
    )
    if st.button("Atualizar dados agora"):
        with st.spinner("Baixando dados..."):
            n_prices, n_divs = update_all(tickers)
        st.success(f"Atualizado! {n_prices} preços e {n_divs} dividendos inseridos/atualizados.")

init_db()
engine = get_engine()

# Seleção principal
col1, col2 = st.columns([3,1])
with col2:
    sel = st.selectbox("Escolha o FII", options=tickers, index=0 if tickers else None)
    
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



if sel:
    with engine.connect() as conn:
        prices = pd.read_sql(text("SELECT * FROM prices WHERE ticker=:t ORDER BY dt"), conn, params={"t": sel})
        divs = pd.read_sql(text("SELECT * FROM dividends WHERE ticker=:t ORDER BY ex_date"), conn, params={"t": sel})
    if prices.empty:
        st.info("Sem dados ainda. Clique em **Atualizar dados agora**.")
    else:
        cot_tab, div_tab, watch_tab, sug_tab = st.tabs(["Cotação", "Dividendos", "Watchlist", "Sugestão"]) 

        with cot_tab:
            # Métricas rápidas (cotação)
            last = prices.iloc[-1]["close"]
            wk = prices.tail(5)["close"].pct_change().dropna().sum()
            mo = prices.tail(22)["close"].pct_change().dropna().sum()
            y = prices.tail(252)["close"].pct_change().dropna().sum()
            m1, m2, m3 = st.columns(3)
            m1.metric("Preço atual", f"R$ {last:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
            m2.metric("Retorno 1M", f"{mo*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X","."))
            m3.metric("Retorno 1A", f"{y*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X","."))

            # Série histórica (linha)
            if isinstance(dias_lookback, int):
                cutoff_sel = pd.to_datetime(prices["dt"]).max() - pd.Timedelta(days=int(dias_lookback))
                prices_plot = prices[pd.to_datetime(prices["dt"]) >= cutoff_sel].copy()
                title_suffix = f" (últimos {dias_lookback} dias)"
            else:
                prices_plot = prices.copy()
                title_suffix = ""

            fig = px.line(prices_plot, x="dt", y="close", title=f"Preço - {sel}{title_suffix}")
            st.plotly_chart(fig, use_container_width=True)

            # Intradiário (cotação)
            with st.expander("Cotação intradiária (hoje) por hora", expanded=True):
                intra = yfd.fetch_intraday_hourly(sel)
                if intra is None or intra.empty:
                    st.info("Sem dados intradiários para hoje.")
                else:
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

        with div_tab:
            if not divs.empty:
                # DY trailing 12m e barra de proventos
                divs["year_month"] = pd.to_datetime(divs["ex_date"]).dt.to_period("M")
                last12 = pd.to_datetime(prices["dt"]).max() - pd.offsets.DateOffset(years=1)
                dyt = divs[pd.to_datetime(divs["ex_date"]) >= last12]["value"].sum() / prices.iloc[-1]["close"]
                st.metric("Dividend Yield (12m, aprox.)", f"{dyt*100:,.2f}%".replace(",", "X").replace(".", ",").replace("X","."))
                bar = px.bar(divs, x="ex_date", y="value", title=f"Proventos - {sel}")
                st.plotly_chart(bar, use_container_width=True)
            else:
                st.info("Sem proventos registrados para este FII.")

            # Top 10 DY (12m) com base na watchlist - abaixo do gráfico de proventos
            st.subheader("Top 10 Dividend Yield (12m, aprox.) — Watchlist")
            try:
                with engine.connect() as conn:
                    last_dt_df = pd.read_sql(text("SELECT MAX(dt) AS last_dt FROM prices"), conn)
                    if last_dt_df.empty or pd.isna(last_dt_df.iloc[0]["last_dt"]):
                        st.info("Sem dados de preços para calcular DY.")
                    else:
                        ref_dt = pd.to_datetime(last_dt_df.iloc[0]["last_dt"]).date()
                        prices_last = pd.read_sql(
                            text(
                                """
                                SELECT p.ticker, p.close
                                FROM prices p
                                JOIN (
                                    SELECT ticker, MAX(dt) AS maxdt
                                    FROM prices
                                    GROUP BY ticker
                                ) m
                                ON p.ticker = m.ticker AND p.dt = m.maxdt
                                """
                            ),
                            conn,
                        )
                        prices_last = prices_last[prices_last["ticker"].isin(tickers)].copy()

                        divs_all = pd.read_sql(text("SELECT ticker, ex_date, value FROM dividends"), conn)
                        if not prices_last.empty and not divs_all.empty:
                            cutoff = pd.Timestamp(ref_dt) - pd.offsets.DateOffset(years=1)
                            divs_all["ex_date"] = pd.to_datetime(divs_all["ex_date"])
                            divs_filt = divs_all[(divs_all["ticker"].isin(tickers)) & (divs_all["ex_date"] >= cutoff)].copy()
                            dy_sum = divs_filt.groupby("ticker")["value"].sum().reset_index(name="div12m")
                            dy_tbl = prices_last.merge(dy_sum, on="ticker", how="left")
                            dy_tbl["div12m"].fillna(0.0, inplace=True)
                            dy_tbl["dy"] = dy_tbl["div12m"] / dy_tbl["close"]
                            dy_tbl = dy_tbl.dropna(subset=["dy"]).copy()
                            top10 = dy_tbl.sort_values("div12m", ascending=False).head(10)
                            if not top10.empty:
                                top10_plot = top10.copy()
                                top10_plot["dy_pct"] = top10_plot["dy"] * 100
                                fig_top10 = px.bar(
                                    top10_plot,
                                    x="dy_pct",
                                    y="ticker",
                                    orientation="h",
                                    title="Top 10 DY (12m, aprox.)",
                                    labels={"dy_pct": "DY %", "ticker": ""},
                                )
                                # Rótulos nas barras e ordenação por valor (desc)
                                fig_top10.update_traces(
                                    text=top10_plot["dy_pct"].map(lambda v: f"{v:.2f}%"),
                                    textposition="outside",
                                )
                                fig_top10.update_layout(
                                    yaxis={
                                        "categoryorder": "array",
                                        "categoryarray": top10_plot["ticker"].tolist(),
                                    },
                                    xaxis_title="DY %",
                                )
                                st.plotly_chart(fig_top10, use_container_width=True)
                            else:
                                st.info("Sem dados suficientes para o Top 10 DY.")
                        else:
                            st.info("Sem dados suficientes para o Top 10 DY.")
            except Exception as e:
                st.warning(f"Não foi possível calcular o Top 10 DY. {e}")

        # Watchlist — Top 5 melhores avaliações (pontuação fixa, sem controles)
        with watch_tab:
            st.subheader("Watchlist — Top 5 melhores avaliações")
            try:
                with engine.connect() as conn:
                    prices_all = pd.read_sql(text("SELECT * FROM prices ORDER BY ticker, dt"), conn)
                    divs_all = pd.read_sql(text("SELECT * FROM dividends"), conn)
                prices_all = prices_all[prices_all["ticker"].isin(tickers)].copy()
                if prices_all.empty:
                    st.info("Sem dados suficientes na watchlist.")
                else:
                    prices_all["dt"] = pd.to_datetime(prices_all["dt"])
                    rows = []
                    for tk in sorted(prices_all["ticker"].unique()):
                        g = prices_all[prices_all["ticker"] == tk].sort_values("dt")
                        if g.empty:
                            continue
                        last_close = float(g.iloc[-1]["close"]) if pd.notna(g.iloc[-1]["close"]) else None
                        last_dt = pd.to_datetime(g["dt"]).max()
                        cutoff_12m = last_dt - pd.offsets.DateOffset(days=365)
                        g12 = g[g["dt"] >= cutoff_12m]
                        if len(g12) >= 2 and last_close is not None and last_close > 0:
                            ret12 = float(g12.iloc[-1]["close"]) / float(g12.iloc[0]["close"]) - 1.0
                        else:
                            ret12 = None
                        ret_series = g["close"].pct_change().dropna().tail(252)
                        vol_ann = float(ret_series.std() * (252 ** 0.5)) if len(ret_series) >= 20 else None
                        vol_21d = float(g["close"].pct_change().dropna().tail(21).std() * (252 ** 0.5)) if g["close"].pct_change().dropna().shape[0] >= 10 else None
                        vol_63d = float(g["close"].pct_change().dropna().tail(63).std() * (252 ** 0.5)) if g["close"].pct_change().dropna().shape[0] >= 20 else None
                        # Retornos recentes
                        if len(g) >= 22 and last_close is not None and pd.notna(g.iloc[-22]["close"]) and g.iloc[-22]["close"]:
                            ret22 = last_close / float(g.iloc[-22]["close"]) - 1.0
                        else:
                            ret22 = None
                        if len(g) >= 63 and last_close is not None and pd.notna(g.iloc[-63]["close"]) and g.iloc[-63]["close"]:
                            ret63 = last_close / float(g.iloc[-63]["close"]) - 1.0
                        else:
                            ret63 = None
                        liq21 = float(pd.to_numeric(g["volume"], errors="coerce").tail(21).mean())
                        dy12 = None
                        dy_cv = None
                        if not divs_all.empty and last_close and last_close > 0:
                            dtk = divs_all[divs_all["ticker"] == tk].copy()
                            if not dtk.empty:
                                dtk["ex_date"] = pd.to_datetime(dtk["ex_date"]) 
                                cutoff_div = last_dt - pd.offsets.DateOffset(years=1)
                                d12 = dtk[dtk["ex_date"] >= cutoff_div]
                                dy12 = float(d12["value"].sum()) / last_close if not d12.empty else 0.0
                                dtk12 = dtk[dtk["ex_date"] >= cutoff_div]
                                if not dtk12.empty:
                                    monthly = dtk12.groupby(dtk12["ex_date"].dt.to_period("M"))["value"].sum()
                                    if monthly.mean() and monthly.mean() != 0:
                                        dy_cv = float(monthly.std() / monthly.mean())
                        # Máximo drawdown (últimos 252 dias)
                        closes_252 = g["close"].tail(252)
                        if len(closes_252) >= 5:
                            roll_max = closes_252.cummax()
                            dd = (closes_252 / roll_max) - 1.0
                            mdd = float(dd.min())
                        else:
                            mdd = None
                        rows.append({
                            "ticker": tk,
                            "preco": last_close,
                            "ret12m": ret12,
                            "vol_ann": vol_ann,
                            "vol21d": vol_21d,
                            "vol63d": vol_63d,
                            "ret22d": ret22,
                            "ret63d": ret63,
                            "liq21d": liq21,
                            "dy12m": dy12,
                            "dy_cv": dy_cv,
                            "mdd": mdd,
                        })

                    mt = pd.DataFrame(rows)
                    if mt.empty:
                        st.info("Sem dados suficientes na watchlist.")
                    else:
                        def norm_minmax(s: pd.Series) -> pd.Series:
                            s = pd.to_numeric(s, errors="coerce")
                            if s.dropna().empty:
                                return pd.Series([0.0] * len(s), index=s.index)
                            vmin, vmax = s.min(), s.max()
                            if pd.isna(vmin) or pd.isna(vmax) or vmax - vmin == 0:
                                return pd.Series([0.5] * len(s), index=s.index)
                            return (s - vmin) / (vmax - vmin)

                        # Normalizações
                        n_dy = norm_minmax(mt["dy12m"]).fillna(0)
                        n_ret12 = norm_minmax(mt["ret12m"]).fillna(0)
                        n_ret22 = norm_minmax(mt["ret22d"]).fillna(0)
                        n_ret63 = norm_minmax(mt["ret63d"]).fillna(0)
                        n_vol_inv = 1 - norm_minmax(mt["vol_ann"]).fillna(0)
                        n_vol21_inv = 1 - norm_minmax(mt["vol21d"]).fillna(0)
                        n_vol63_inv = 1 - norm_minmax(mt["vol63d"]).fillna(0)
                        n_liq = norm_minmax(mt["liq21d"]).fillna(0)
                        n_stb = 1 - norm_minmax(mt["dy_cv"]).fillna(0)
                        # Drawdown: magnitude (positivo) e invertido (menor melhor)
                        dd_mag = mt["mdd"].apply(lambda v: -float(v) if pd.notna(v) and v < 0 else 0.0)
                        n_ddinv = 1 - norm_minmax(dd_mag).fillna(0)

                        # Scores por horizonte
                        mt["score_curto"] = (
                            0.45 * n_ret22 + 0.25 * n_vol21_inv + 0.25 * n_liq + 0.05 * n_dy
                        )
                        mt["score_medio"] = (
                            0.25 * n_ret63 + 0.15 * n_ret12 + 0.20 * n_vol63_inv + 0.20 * n_dy + 0.15 * n_stb + 0.05 * n_liq
                        )
                        mt["score_longo"] = (
                            0.35 * n_dy + 0.25 * n_stb + 0.15 * n_vol_inv + 0.15 * n_ddinv + 0.10 * n_ret12
                        )

                        # Selecionar 5 FIIs distintos
                        remaining = mt.copy()
                        winners = []
                        # Curto
                        if not remaining["score_curto"].dropna().empty:
                            i1 = remaining["score_curto"].idxmax()
                            winners.append(("Curto prazo", remaining.loc[i1]))
                            remaining = remaining[remaining["ticker"] != remaining.loc[i1, "ticker"]]
                        # Médio
                        if not remaining.empty and not remaining["score_medio"].dropna().empty:
                            i2 = remaining["score_medio"].idxmax()
                            winners.append(("Médio prazo", remaining.loc[i2]))
                            remaining = remaining[remaining["ticker"] != remaining.loc[i2, "ticker"]]
                        # Longo
                        if not remaining.empty and not remaining["score_longo"].dropna().empty:
                            i3 = remaining["score_longo"].idxmax()
                            winners.append(("Longo prazo", remaining.loc[i3]))
                            remaining = remaining[remaining["ticker"] != remaining.loc[i3, "ticker"]]

                        # Extras (2) pelo melhor score entre curto/médio/longo — com rotulagem por horizonte
                        if not remaining.empty:
                            best_combo = remaining[["score_curto", "score_medio", "score_longo"]].max(axis=1)
                            best_idx = best_combo.dropna().sort_values(ascending=False).head(2).index.tolist()
                            for bx in best_idx:
                                r = remaining.loc[bx]
                                sc_c = float(r.get("score_curto", 0) or 0)
                                sc_m = float(r.get("score_medio", 0) or 0)
                                sc_l = float(r.get("score_longo", 0) or 0)
                                if sc_c >= sc_m and sc_c >= sc_l:
                                    label = "Curto prazo"
                                elif sc_m >= sc_c and sc_m >= sc_l:
                                    label = "Médio prazo"
                                else:
                                    label = "Longo prazo"
                                winners.append((label, r))
                            remaining = remaining.drop(index=best_idx, errors="ignore")

                        # Apresentação
                        for horizon, row in winners:
                            tk = row["ticker"]
                            if horizon == "Curto prazo":
                                score_pct = float(row["score_curto"]) * 100 if pd.notna(row["score_curto"]) else 0
                                contrib = {
                                    "Retorno 22 dias": 0.45 * float(n_ret22[mt["ticker"] == tk].iloc[0]) if not n_ret22.empty else 0,
                                    "Baixa volatilidade (21d)": 0.25 * float(n_vol21_inv[mt["ticker"] == tk].iloc[0]) if not n_vol21_inv.empty else 0,
                                    "Liquidez": 0.25 * float(n_liq[mt["ticker"] == tk].iloc[0]) if not n_liq.empty else 0,
                                    "Dividend Yield": 0.05 * float(n_dy[mt["ticker"] == tk].iloc[0]) if not n_dy.empty else 0,
                                }
                            elif horizon == "Médio prazo":
                                score_pct = float(row["score_medio"]) * 100 if pd.notna(row["score_medio"]) else 0
                                contrib = {
                                    "Retorno 63 dias": 0.25 * float(n_ret63[mt["ticker"] == tk].iloc[0]) if not n_ret63.empty else 0,
                                    "Retorno 12 meses": 0.15 * float(n_ret12[mt["ticker"] == tk].iloc[0]) if not n_ret12.empty else 0,
                                    "Baixa volatilidade (63d)": 0.20 * float(n_vol63_inv[mt["ticker"] == tk].iloc[0]) if not n_vol63_inv.empty else 0,
                                    "Dividend Yield": 0.20 * float(n_dy[mt["ticker"] == tk].iloc[0]) if not n_dy.empty else 0,
                                    "Estabilidade do DY": 0.15 * float(n_stb[mt["ticker"] == tk].iloc[0]) if not n_stb.empty else 0,
                                    "Liquidez": 0.05 * float(n_liq[mt["ticker"] == tk].iloc[0]) if not n_liq.empty else 0,
                                }
                            elif horizon == "Longo prazo":
                                score_pct = float(row["score_longo"]) * 100 if pd.notna(row["score_longo"]) else 0
                                contrib = {
                                    "Dividend Yield": 0.35 * float(n_dy[mt["ticker"] == tk].iloc[0]) if not n_dy.empty else 0,
                                    "Estabilidade do DY": 0.25 * float(n_stb[mt["ticker"] == tk].iloc[0]) if not n_stb.empty else 0,
                                    "Baixa volatilidade (252d)": 0.15 * float(n_vol_inv[mt["ticker"] == tk].iloc[0]) if not n_vol_inv.empty else 0,
                                    "Menor drawdown": 0.15 * float(n_ddinv[mt["ticker"] == tk].iloc[0]) if not n_ddinv.empty else 0,
                                    "Retorno 12 meses": 0.10 * float(n_ret12[mt["ticker"] == tk].iloc[0]) if not n_ret12.empty else 0,
                                }
                            else:
                                # Não deve ocorrer; winners só contém rótulos "Curto prazo", "Médio prazo" ou "Longo prazo"
                                score_pct = 0
                                contrib = {}

                            total_c = sum(contrib.values()) if contrib else 0.0
                            factors = []
                            if total_c > 0:
                                factors = [
                                    (name, (val/total_c)*100.0)
                                    for name, val in sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:3]
                                ]

                            st.markdown(f"### {horizon}: {tk} — Pontuação {score_pct:.1f}%")
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("Preço atual", f"R$ {row['preco']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                                st.metric("Dividend Yield (12m)", f"{((row['dy12m'] or 0)*100 if row['dy12m'] is not None else 0):,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
                            with c2:
                                ret12p = (row['ret12m'] or 0) * 100 if row['ret12m'] is not None else 0
                                st.metric("Retorno 12m", f"{ret12p:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
                                volp = (row['vol_ann'] or 0) * 100 if row['vol_ann'] is not None else 0
                                st.metric("Volatilidade anualizada", f"{volp:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
                            with c3:
                                st.metric("Liquidez média (21d)", f"{row['liq21d']:,.0f}".replace(",", "."))
                                cvv = row['dy_cv'] if row['dy_cv'] is not None else 0
                                st.metric("Estabilidade do DY (CV)", f"{cvv:.2f}")

                            if factors:
                                bullets = "\n".join([f"- {name}: {share:.1f}% da pontuação" for name, share in factors])
                                st.markdown("Por que sugerimos este FII:\n" + bullets)
                            else:
                                st.markdown("Por que sugerimos este FII:\n- Dados insuficientes para detalhar a pontuação")
                            st.divider()
            except Exception as e:
                st.info(f"Não foi possível montar a seção Watchlist. {e}")

        # Aba Sugestão — somente sugestões trimestrais
        with sug_tab:

            # Sugestões — Top 5 custo-benefício (trimestre)
            st.subheader("Sugestões — Top 5 custo-benefício (trimestre)")
            try:
                with engine.connect() as conn:
                    prices_all = pd.read_sql(text("SELECT * FROM prices ORDER BY ticker, dt"), conn)
                    divs_all = pd.read_sql(text("SELECT * FROM dividends"), conn)
                prices_all = prices_all[prices_all["ticker"].isin(tickers)].copy()
                if prices_all.empty:
                    st.info("Sem dados suficientes na watchlist.")
                else:
                    prices_all["dt"] = pd.to_datetime(prices_all["dt"]) 
                    rows = []
                    for tk in sorted(prices_all["ticker"].unique()):
                        g = prices_all[prices_all["ticker"] == tk].sort_values("dt")
                        if g.empty:
                            continue
                        last_close = float(g.iloc[-1]["close"]) if pd.notna(g.iloc[-1]["close"]) else None
                        last_dt = pd.to_datetime(g["dt"]).max()
                        # Retorno 63 dias (aprox. 3 meses) e vol 63d
                        if len(g) >= 63 and last_close is not None and pd.notna(g.iloc[-63]["close"]) and g.iloc[-63]["close"]:
                            ret63 = last_close / float(g.iloc[-63]["close"]) - 1.0
                        else:
                            ret63 = None
                        ret_series = g["close"].pct_change().dropna().tail(63)
                        vol63 = float(ret_series.std()) if len(ret_series) >= 10 else None
                        # DY 3m
                        dy3m = None
                        if not divs_all.empty and last_close and last_close > 0:
                            dtk = divs_all[divs_all["ticker"] == tk].copy()
                            if not dtk.empty:
                                dtk["ex_date"] = pd.to_datetime(dtk["ex_date"]) 
                                cutoff_3m = last_dt - pd.offsets.DateOffset(months=3)
                                d3 = dtk[dtk["ex_date"] >= cutoff_3m]
                                dy3m = float(d3["value"].sum()) / last_close if not d3.empty else 0.0
                        liq21 = float(pd.to_numeric(g["volume"], errors="coerce").tail(21).mean())
                        rows.append({
                            "ticker": tk,
                            "preco": last_close,
                            "ret63": ret63,
                            "vol63": vol63,
                            "dy3m": dy3m,
                            "liq21d": liq21,
                        })
                    mtq = pd.DataFrame(rows)
                    if mtq.empty:
                        st.info("Sem dados suficientes na watchlist.")
                    else:
                        def norm_minmax_q(s: pd.Series) -> pd.Series:
                            s = pd.to_numeric(s, errors="coerce")
                            if s.dropna().empty:
                                return pd.Series([0.0] * len(s), index=s.index)
                            vmin, vmax = s.min(), s.max()
                            if pd.isna(vmin) or pd.isna(vmax) or vmax - vmin == 0:
                                return pd.Series([0.5] * len(s), index=s.index)
                            return (s - vmin) / (vmax - vmin)
                        # Ratio retorno/vol 63d
                        ratio = []
                        for _, r in mtq.iterrows():
                            rr = r.get("ret63")
                            vv = r.get("vol63")
                            ratio.append((rr / (vv if (vv is not None and vv != 0) else 1e-9)) if rr is not None and vv is not None else None)
                        mtq["ratio"] = ratio
                        n_ratio = norm_minmax_q(mtq["ratio"]).fillna(0)
                        n_ret63 = norm_minmax_q(mtq["ret63"]).fillna(0)
                        n_vol63_inv = 1 - norm_minmax_q(mtq["vol63"]).fillna(0)
                        n_dy3m = norm_minmax_q(mtq["dy3m"]).fillna(0)
                        n_liq = norm_minmax_q(mtq["liq21d"]).fillna(0)
                        # Pesos custo-benefício (trimestre)
                        w_ratio, w_ret, w_vol, w_dy, w_liq = 0.45, 0.25, 0.15, 0.10, 0.05
                        mtq["score_q"] = (
                            w_ratio*n_ratio + w_ret*n_ret63 + w_vol*n_vol63_inv + w_dy*n_dy3m + w_liq*n_liq
                        )
                        top5q = mtq.sort_values("score_q", ascending=False).head(5).reset_index(drop=True)
                        for _, r in top5q.iterrows():
                            tk = r["ticker"]
                            score_pct = (r["score_q"] if pd.notna(r["score_q"]) else 0) * 100
                            contrib = {
                                "Retorno/Vol 63d": w_ratio * float(n_ratio[mtq["ticker"] == tk].iloc[0]) if not n_ratio.empty else 0,
                                "Retorno 63d": w_ret * float(n_ret63[mtq["ticker"] == tk].iloc[0]) if not n_ret63.empty else 0,
                                "Baixa volatilidade 63d": w_vol * float(n_vol63_inv[mtq["ticker"] == tk].iloc[0]) if not n_vol63_inv.empty else 0,
                                "Dividendos 3m": w_dy * float(n_dy3m[mtq["ticker"] == tk].iloc[0]) if not n_dy3m.empty else 0,
                                "Liquidez": w_liq * float(n_liq[mtq["ticker"] == tk].iloc[0]) if not n_liq.empty else 0,
                            }
                            total_c = sum(contrib.values()) if contrib else 0.0
                            factors = []
                            if total_c > 0:
                                factors = [
                                    (name, (val/total_c)*100.0)
                                    for name, val in sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:3]
                                ]
                            st.markdown(f"### {tk} — Pontuação {score_pct:.1f}% (custo-benefício trimestral)")
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("Preço atual", f"R$ {r['preco']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
                                ret63p = (r['ret63'] or 0) * 100 if r['ret63'] is not None else 0
                                st.metric("Retorno 63d", f"{ret63p:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
                            with c2:
                                vol63p = (r['vol63'] or 0) * 100 if r['vol63'] is not None else 0
                                st.metric("Volatilidade 63d (dp)", f"{vol63p:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
                                dy3mp = (r['dy3m'] or 0) * 100 if r['dy3m'] is not None else 0
                                st.metric("Dividendos 3m / preço", f"{dy3mp:,.2f}%".replace(",", "X").replace(".", ",").replace("X", "."))
                            with c3:
                                st.metric("Liquidez média (21d)", f"{r['liq21d']:,.0f}".replace(",", "."))
                            if factors:
                                bullets = "\n".join([f"- {name}: {share:.1f}% da pontuação" for name, share in factors])
                                st.markdown("Por que está entre as melhores (trimestre):\n" + bullets)
                            st.divider()
            except Exception:
                st.info("Não foi possível montar as sugestões trimestrais.")


st.caption("Dados via Yahoo Finance (yfinance). DY e métricas são estimativas e não substituem análises oficiais.")
