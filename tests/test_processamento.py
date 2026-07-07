import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.processamento import (
    build_feature_frame,
    carregar_e_dividir,
    montar_painel_long,
    montar_painel_wide,
    split_treino_teste,
)


def test_painel_wide_sem_valores_faltantes_nas_colunas_de_serie():
    painel = montar_painel_wide()
    colunas_serie = [c for c in painel.columns if c != "data"]
    assert painel[colunas_serie].isna().sum().sum() == 0
    assert len(painel) > 100  # mais de ~8 anos de historico mensal


def test_painel_long_tem_tres_segmentos_e_alvo_binario():
    painel = montar_painel_long()
    assert set(painel["segmento"].unique()) == {"total", "pf", "pj"}
    assert set(painel["risco_subida"].unique()) <= {0, 1}
    assert painel.isna().sum().sum() == 0


def test_build_feature_frame_e_totalmente_numerico():
    painel = montar_painel_long()
    X, colunas = build_feature_frame(painel)
    assert X.shape[0] == len(painel)
    assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)
    assert "data" not in colunas
    assert "risco_subida" not in colunas
    assert "segmento_total" in colunas


def test_split_e_cronologico_sem_vazamento_de_futuro():
    painel = montar_painel_long()
    treino, teste = split_treino_teste(painel, frac_teste=0.2)

    assert treino["data"].max() < teste["data"].min()
    assert len(treino) > 0 and len(teste) > 0


def test_carregar_e_dividir_colunas_consistentes_entre_treino_e_teste():
    X_treino, X_teste, y_treino, y_teste, colunas, treino_df, teste_df = carregar_e_dividir()

    assert list(X_treino.columns) == list(X_teste.columns) == colunas
    assert len(X_treino) == len(y_treino)
    assert len(X_teste) == len(y_teste)
