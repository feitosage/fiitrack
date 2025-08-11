
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "fii_quotes.db")
DEFAULT_WATCHLIST = [
    t.strip()
    for t in os.getenv(
        "WATCHLIST",
        "MXRF11,MCRE11,VGHF11,VISC11,RURA11,TRXF11,XPLG11,HCTR11,RECT11,RZTR11",
    ).split(",")
    if t.strip()
]
