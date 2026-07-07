"""
processamento.py
-----------------
Junta as series do Banco Central num painel mensal e constroi as features
e o alvo do modelo de risco de credito.

Pergunta de negocio: "dado o cenario macroeconomico atual, a taxa de
inadimplencia de um determinado segmento de credito (total / pessoa fisica /
pessoa juridica) vai SUBIR no proximo mes?"

Isso e o tipo de sinal que uma area de risco de credito (banco, fintech,
financeira) usa para antecipar politica de concessao de credito, provisao
para devedores duvidosos (PDD) e apetite de risco.

Formato dos dados: painel "long" (uma linha por mes x segmento), o que
triplica o numero de amostras em relacao a usar so a serie total - com
series macro tao curtas (mensais), isso ajuda a ter uma base de treino
minimamente razoavel.

Este modulo e importado tanto por treino.py (batch) quanto por
api/main.py e app/dashboard.py (online), para garantir que a mesma
transformacao seja usada em treino e em producao.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
FEATURE_LIST_PATH = Path("models/feature_list.json")

SEGMENTOS = {
    "total": "bcb_21082_inadimplencia_total.json",
    "pf": "bcb_21112_inadimplencia_pf.json",
    "pj": "bcb_21086_inadimplencia_pj.json",
}
MACRO = {
    "selic_mensal": "bcb_4390_selic_mensal.json",
    "ipca_mensal": "bcb_433_ipca_mensal.json",
    "desemprego": "bcb_24369_desemprego.json",
    "saldo_credito": "bcb_20542_saldo_credito.json",
}

TARGET = "risco_subida"
N_LAGS = 3


def _carregar_serie(path: Path, nome_coluna: str) -> pd.DataFrame:
    dados = json.loads(path.read_text())
    df = pd.DataFrame(dados)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df[nome_coluna] = pd.to_numeric(df["valor"], errors="coerce")
    return df[["data", nome_coluna]]


def montar_painel_wide() -> pd.DataFrame:
    """Monta um DataFrame mensal (uma linha por mes) com todas as series."""
    df = None
    for nome, arquivo in {**SEGMENTOS, **MACRO}.items():
        col_name = nome if nome in MACRO else f"inad_{nome}"
        serie = _carregar_serie(RAW_DIR / arquivo, col_name)
        df = serie if df is None else df.merge(serie, on="data", how="inner")

    df = df.sort_values("data").reset_index(drop=True)
    return df


def _engenharia_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mes"] = df["data"].dt.month
    df["trimestre"] = df["data"].dt.quarter
    df["tendencia"] = np.arange(len(df))  # nro de meses desde o inicio da serie

    # Selic acumulada nos ultimos 3 meses (custo do dinheiro recente)
    df["selic_acum_3m"] = df["selic_mensal"].rolling(3).sum()
    # IPCA acumulado nos ultimos 12 meses (proxy de inflacao anual)
    df["ipca_acum_12m"] = df["ipca_mensal"].rolling(12).sum()
    # variacao percentual do saldo de credito (expansao/retracao do credito)
    df["saldo_credito_var_mensal"] = df["saldo_credito"].pct_change() * 100
    df["saldo_credito_var_12m"] = df["saldo_credito"].pct_change(12) * 100
    return df


def montar_painel_long() -> pd.DataFrame:
    """Painel long: uma linha por (mes, segmento), com lags do proprio
    segmento e o alvo binario 'risco_subida' (a inadimplencia deste
    segmento sobe no mes seguinte?)."""
    wide = _engenharia_features(montar_painel_wide())

    linhas = []
    for segmento in SEGMENTOS:
        col = f"inad_{segmento}"
        bloco = wide[[
            "data", "mes", "trimestre", "tendencia",
            "selic_mensal", "selic_acum_3m", "ipca_mensal", "ipca_acum_12m",
            "desemprego", "saldo_credito_var_mensal", "saldo_credito_var_12m",
            col,
        ]].copy()
        bloco = bloco.rename(columns={col: "inadimplencia"})
        bloco["segmento"] = segmento

        for lag in range(1, N_LAGS + 1):
            bloco[f"inadimplencia_lag{lag}"] = bloco["inadimplencia"].shift(lag)

        bloco[TARGET] = (bloco["inadimplencia"].shift(-1) > bloco["inadimplencia"]).astype("Int64")

        linhas.append(bloco)

    painel = pd.concat(linhas, ignore_index=True)
    painel = painel.dropna().reset_index(drop=True)
    painel[TARGET] = painel[TARGET].astype(int)
    return painel


def build_feature_frame(df: pd.DataFrame, fit_columns=None):
    """Recebe um painel (wide ou parcial) e retorna (X, colunas) prontas
    para o modelo - one-hot no segmento, sem a coluna alvo/data."""
    df = df.copy()
    X = pd.get_dummies(df, columns=["segmento"], drop_first=False)

    drop_cols = [c for c in ["data", "inadimplencia", TARGET] if c in X.columns]
    X = X.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number, bool]).astype(float)

    if fit_columns is not None:
        X = X.reindex(columns=fit_columns, fill_value=0.0)

    return X, list(X.columns)


def split_treino_teste(painel: pd.DataFrame, frac_teste: float = 0.2):
    """Split cronologico (nao aleatorio!): treina no passado, testa no
    periodo mais recente - a forma correta de validar séries temporais,
    evitando "vazamento do futuro" para o passado."""
    datas_unicas = sorted(painel["data"].unique())
    corte_idx = int(len(datas_unicas) * (1 - frac_teste))
    data_corte = datas_unicas[corte_idx]

    treino = painel[painel["data"] < data_corte].reset_index(drop=True)
    teste = painel[painel["data"] >= data_corte].reset_index(drop=True)
    return treino, teste


def carregar_e_dividir(frac_teste: float = 0.2):
    painel = montar_painel_long()
    treino, teste = split_treino_teste(painel, frac_teste)

    y_treino = treino[TARGET]
    y_teste = teste[TARGET]

    X_treino, colunas = build_feature_frame(treino)
    X_teste, _ = build_feature_frame(teste, fit_columns=colunas)

    return X_treino, X_teste, y_treino, y_teste, colunas, treino, teste


def salvar_processados(X_treino, X_teste, y_treino, y_teste, colunas):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    X_treino.assign(**{TARGET: y_treino.values}).to_csv(PROCESSED_DIR / "treino.csv", index=False)
    X_teste.assign(**{TARGET: y_teste.values}).to_csv(PROCESSED_DIR / "teste.csv", index=False)
    FEATURE_LIST_PATH.write_text(json.dumps(colunas, indent=2))


if __name__ == "__main__":
    X_treino, X_teste, y_treino, y_teste, colunas, treino, teste = carregar_e_dividir()
    salvar_processados(X_treino, X_teste, y_treino, y_teste, colunas)
    print(f"Painel: {len(treino) + len(teste)} linhas (treino={len(X_treino)}, teste={len(X_teste)})")
    print(f"Periodo treino: {treino['data'].min().date()} a {treino['data'].max().date()}")
    print(f"Periodo teste:  {teste['data'].min().date()} a {teste['data'].max().date()}")
    print(f"Taxa de 'sobe' no treino: {y_treino.mean():.1%} | no teste: {y_teste.mean():.1%}")
    print(f"Numero de features: {len(colunas)}")
