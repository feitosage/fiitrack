#!/usr/bin/env python3
import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"


def find_free_port(start: int = 8501, end: int = 8600) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Nenhuma porta livre encontrada no intervalo especificado.")


def ensure_venv() -> Path:
    if not VENV_DIR.exists():
        print("[bootstrap] Criando ambiente virtual em .venv ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print("[bootstrap] Ambiente virtual .venv já existe.")
    python_bin = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python_bin.exists():
        raise RuntimeError("Python da venv não encontrado. Remova .venv e tente novamente.")
    return python_bin


def pip_install(python_bin: Path, requirements: Path) -> None:
    print("[bootstrap] Atualizando pip/setuptools/wheel ...")
    subprocess.check_call([str(python_bin), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]) 
    print("[bootstrap] Instalando dependências de requirements.txt ...")
    subprocess.check_call([str(python_bin), "-m", "pip", "install", "-r", str(requirements)])


def run_streamlit(python_bin: Path, port: int, headless: bool = True) -> int:
    env = os.environ.copy()
    args = [
        str(python_bin),
        "-m",
        "streamlit",
        "run",
        str(REPO_ROOT / "app" / "streamlit_app.py"),
        "--server.port",
        str(port),
    ]
    if headless:
        args += ["--server.headless", "true"]
    print(f"[bootstrap] Iniciando Streamlit em http://localhost:{port} ...")
    # Usa Popen para anexar ao terminal atual
    return subprocess.call(args, env=env, cwd=str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap do FII Quote Tracker")
    p.add_argument(
        "--port",
        type=int,
        default=0,
        help="Porta preferida/inicial (se ocupada, usa a próxima livre; 0 = escolher automaticamente a partir de 8501)",
    )
    p.add_argument("--no-install", action="store_true", help="Não instalar/atualizar dependências")
    p.add_argument("--headless", action="store_true", help="Rodar em modo headless")
    return p.parse_args()


def main() -> None:
    if sys.version_info < (3, 10):
        print("[bootstrap] Requer Python 3.10+.")
        sys.exit(1)

    args = parse_args()
    # Escolhe porta livre: se --port fornecida, tenta a partir dela; caso contrário, inicia em 8501
    start_port = args.port if args.port and args.port > 0 else 8501
    # Primeiro, escolhe uma porta livre como tentativa inicial
    port = find_free_port(start=start_port, end=8600)
    python_bin = ensure_venv()

    if not args.no_install:
        req = REPO_ROOT / "requirements.txt"
        if not req.exists():
            print("[bootstrap] requirements.txt não encontrado!")
            sys.exit(1)
        pip_install(python_bin, req)

    # Tenta iniciar o Streamlit; se falhar (ex.: porta ocupada na hora do bind), tenta portas subsequentes
    for candidate_port in range(port, 8601):
        code = run_streamlit(python_bin, port=candidate_port, headless=args.headless)
        if code == 0:
            # Processo terminou normalmente (fim do app)
            sys.exit(0)
        else:
            # Falhou rapidamente (p.ex., porta em uso). Tenta próxima porta.
            print(f"[bootstrap] Falha ao iniciar na porta {candidate_port} (código {code}). Tentando próxima...")
            continue
    print("[bootstrap] Não foi possível iniciar o Streamlit em nenhuma porta entre {port} e 8600.")
    sys.exit(1)


if __name__ == "__main__":
    main()



