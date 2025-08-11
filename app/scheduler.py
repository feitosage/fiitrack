
from apscheduler.schedulers.background import BackgroundScheduler
from app.ingest import update_all
from app.settings import DEFAULT_WATCHLIST

def start_scheduler():
    sched = BackgroundScheduler(timezone="America/Sao_Paulo")
    # Atualiza a cada dia útil às 20:00 BRT (aprox. após fechamento / ajustes)
    sched.add_job(lambda: update_all(DEFAULT_WATCHLIST), "cron", day_of_week="mon-fri", hour=20, minute=0)
    sched.start()
    return sched
