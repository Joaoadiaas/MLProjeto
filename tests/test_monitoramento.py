import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.monitoramento import calcular_relatorio_drift, _psi_da_coluna


def test_psi_e_zero_para_distribuicoes_identicas():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=1000))
    psi = _psi_da_coluna(s, s)
    assert psi < 1e-6


def test_psi_sinaliza_distribuicao_deslocada():
    rng = np.random.default_rng(0)
    referencia = pd.Series(rng.normal(loc=0, scale=1, size=2000))
    deslocada = pd.Series(rng.normal(loc=3, scale=1, size=2000))
    psi = _psi_da_coluna(referencia, deslocada)
    assert psi > 0.2


def test_relatorio_drift_sinaliza_apenas_coluna_deslocada():
    rng = np.random.default_rng(0)
    referencia = pd.DataFrame({
        "feature_estavel": rng.normal(size=1000),
        "feature_com_drift": rng.normal(size=1000),
        "risco_subida": rng.integers(0, 2, size=1000),
    })
    atual = referencia.copy()
    atual["feature_com_drift"] = rng.normal(loc=5, size=1000)

    relatorio = calcular_relatorio_drift(referencia, atual)
    assert "feature_com_drift" in relatorio["features_sinalizadas"]
    assert "feature_estavel" not in relatorio["features_sinalizadas"]
    assert "risco_subida" not in relatorio["features"]
    assert relatorio["retreino_recomendado"] is True
