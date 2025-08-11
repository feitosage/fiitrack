
# FII Quote Tracker (Python + Streamlit)

Aplicação intermediária para acompanhar cotações de Fundos Imobiliários (FIIs) da B3.
Tecnologias: Python, Streamlit, SQLite, APScheduler e `yfinance` (ou BRAPI opcional).

## Funcionalidades
- Watchlist de FIIs (ex: HGLG11, KNRI11, MXRF11)
- Atualização de preços (manual no app e agendada via APScheduler)
- Banco local SQLite para histórico
- Dashboard com gráfico de preço, retorno e dividend yield histórico (se disponível)
- Alerts simples por preço (e-mail opcional)

## Como rodar
Opção 1 (recomendada): bootstrap automatizado
```bash
python bootstrap.py --headless
```

Opção 2 (manual)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # edite se quiser e-mail para alertas
streamlit run app/streamlit_app.py
```

## Fontes de dados
Por padrão usamos `yfinance` com tickers formados por `TICKER.SA` (ex: `HGLG11.SA`). 
Você pode alternar para BRAPI (https://brapi.dev) em `app/data_sources/brapi.py` se tiver uma API key.

## Escalonamento
Para produção, recomenda-se rodar o job de atualização com um scheduler externo (ex: cron, systemd ou GitHub Actions) e um banco como Postgres/TimescaleDB.
