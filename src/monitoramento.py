"""
monitoramento.py
-----------------
Monitoramento de drift de dados usando o PSI (Population Stability Index),
a metrica padrao que times de risco de credito usam para decidir quando um
modelo precisa ser retreinado - sem dependencias alem de numpy/pandas.

Faz muito sentido aqui: o Brasil passou por cenarios macroeconomicos bem
diferentes entre 2011-2026 (Selic a 2% ao ano em 2020, Selic acima de 13%
em outros periodos) - um modelo treinado num regime de juros baixo pode
nao generalizar bem para um regime de juros alto. Este script detecta
esse tipo de mudanca de cenario.

Interpretacao do PSI (regra pratica do mercado):
    < 0.1   -> sem mudanca significativa
    0.1-0.2 -> mudanca moderada, ficar de olho
    > 0.2   -> mudanca significativa, investigar/retreinar

Uso:
    python src/monitoramento.py --referencia data/processed/treino.csv \
                                 --atual data/processed/teste.csv \
                                 --saida models/relatorio_drift.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _psi_da_coluna(referencia: pd.Series, atual: pd.Series, bins: int = 10) -> float:
    referencia = referencia.dropna()
    atual = atual.dropna()

    if referencia.nunique() <= bins:
        categorias = sorted(set(referencia.unique()) | set(atual.unique()))
        ref_prop = referencia.value_counts(normalize=True).reindex(categorias, fill_value=0)
        atual_prop = atual.value_counts(normalize=True).reindex(categorias, fill_value=0)
    else:
        bordas = np.quantile(referencia, np.linspace(0, 1, bins + 1))
        bordas = np.unique(bordas)
        if len(bordas) < 2:
            return 0.0
        ref_bin = pd.cut(referencia, bins=bordas, include_lowest=True)
        atual_bin = pd.cut(atual, bins=bordas, include_lowest=True)
        ref_prop = ref_bin.value_counts(normalize=True, sort=False)
        atual_prop = atual_bin.value_counts(normalize=True, sort=False)

    eps = 1e-4
    ref_pct = ref_prop.values + eps
    atual_pct = atual_prop.reindex(ref_prop.index, fill_value=0).values + eps

    psi = np.sum((atual_pct - ref_pct) * np.log(atual_pct / ref_pct))
    return float(psi)


def calcular_relatorio_drift(referencia: pd.DataFrame, atual: pd.DataFrame, excluir=("risco_subida",)) -> dict:
    colunas = [c for c in referencia.columns if c in atual.columns and c not in excluir]
    relatorio = {}
    for col in colunas:
        psi = _psi_da_coluna(referencia[col], atual[col])
        if psi < 0.1:
            status = "estavel"
        elif psi < 0.2:
            status = "drift_moderado"
        else:
            status = "drift_significativo"
        relatorio[col] = {"psi": round(psi, 4), "status": status}

    sinalizadas = [c for c, v in relatorio.items() if v["status"] != "estavel"]
    return {
        "features": relatorio,
        "features_sinalizadas": sinalizadas,
        "retreino_recomendado": len(sinalizadas) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--referencia", default="data/processed/treino.csv")
    parser.add_argument("--atual", default="data/processed/teste.csv")
    parser.add_argument("--saida", default="models/relatorio_drift.json")
    args = parser.parse_args()

    referencia = pd.read_csv(args.referencia)
    atual = pd.read_csv(args.atual)
    relatorio = calcular_relatorio_drift(referencia, atual)

    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))

    print(f"{len(relatorio['features'])} features verificadas.")
    if relatorio["features_sinalizadas"]:
        print(f"Drift sinalizado em: {relatorio['features_sinalizadas']}")
    else:
        print("Nenhum drift significativo detectado.")
    print(f"Relatorio salvo em {args.saida}")


if __name__ == "__main__":
    main()
