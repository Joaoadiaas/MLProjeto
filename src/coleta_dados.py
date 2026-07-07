"""
coleta_dados.py
----------------
Coleta series historicas REAIS do Sistema Gerenciador de Series Temporais
(SGS) do Banco Central do Brasil - a mesma API que bancos e fintechs usam
para acompanhar indicadores de credito e macroeconomia no Brasil.

Series utilizadas:
    21082 - Inadimplencia da carteira de credito total (SFN)
    21112 - Inadimplencia da carteira de credito - Pessoas Fisicas
    21086 - Inadimplencia da carteira de credito - Pessoas Juridicas
    4390  - Taxa Selic acumulada no mes
    433   - IPCA - variacao mensal
    24369 - Taxa de desocupacao (PNAD Continua)
    20542 - Saldo da carteira de credito com recursos livres (R$ milhoes)

Por padrao, este script baixa os dados diretamente da API publica do BCB
(https://api.bcb.gov.br) - basta ter internet. Uma copia dos dados usada
para construir este projeto fica salva em data/raw/*.json para garantir
que o pipeline funcione mesmo offline ou se a API estiver instavel.

Uso:
    python src/coleta_dados.py                  # baixa os dados mais recentes
    python src/coleta_dados.py --offline         # usa os JSONs ja salvos em data/raw/
"""
import argparse
import json
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")

SERIES = {
    "21082": "bcb_21082_inadimplencia_total.json",
    "21112": "bcb_21112_inadimplencia_pf.json",
    "21086": "bcb_21086_inadimplencia_pj.json",
    "4390": "bcb_4390_selic_mensal.json",
    "433": "bcb_433_ipca_mensal.json",
    "24369": "bcb_24369_desemprego.json",
    "20542": "bcb_20542_saldo_credito.json",
}

BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"


def baixar_serie(codigo: str, data_inicial: str = "01/01/2011") -> list:
    url = f"{BASE_URL.format(codigo=codigo)}?formato=json&dataInicial={data_inicial}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true",
        help="nao acessa a internet - apenas valida os arquivos ja salvos em data/raw/",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for codigo, filename in SERIES.items():
        path = RAW_DIR / filename
        if args.offline:
            if not path.exists():
                raise FileNotFoundError(f"{path} nao existe. Rode sem --offline para baixar.")
            dados = json.loads(path.read_text())
            print(f"[offline] serie {codigo}: {len(dados)} observacoes em {path}")
            continue

        try:
            dados = baixar_serie(codigo)
            path.write_text(json.dumps(dados))
            print(f"serie {codigo} -> {len(dados)} observacoes salvas em {path}")
        except requests.RequestException as e:
            print(f"AVISO: falha ao baixar serie {codigo} ({e}). Mantendo arquivo local existente.")


if __name__ == "__main__":
    main()
