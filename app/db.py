
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from app.settings import DB_PATH

def get_engine() -> Engine:
    return create_engine(f"sqlite:///{DB_PATH}", future=True)

def init_db():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                dt DATE NOT NULL,
                close REAL NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                volume REAL,
                UNIQUE(ticker, dt)
            );
        '''))
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS dividends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                ex_date DATE NOT NULL,
                value REAL NOT NULL,
                UNIQUE(ticker, ex_date)
            );
        '''))
    return engine
