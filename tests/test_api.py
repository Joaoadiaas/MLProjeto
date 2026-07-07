"""
Testes da API. Os testes que dependem do modelo treinado sao pulados
automaticamente se `models/modelo_risco_credito.pkl` ainda nao existir
(rode `python src/treino.py` antes).
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from api.main import app  # noqa: E402
from api.schemas import CenarioMacro  # noqa: E402

MODELO_EXISTE = (Path(__file__).resolve().parent.parent / "models" / "modelo_risco_credito.pkl").exists()

CENARIO_EXEMPLO = CenarioMacro.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client():
    # Usar o TestClient como context manager dispara o lifespan
    # (startup/shutdown) da aplicacao - sem isso, o modelo nunca carrega
    # e toda chamada retornaria 503.
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


@pytest.mark.skipif(not MODELO_EXISTE, reason="modelo ainda nao treinado - rode src/treino.py")
def test_prever_retorna_probabilidade_valida(client):
    response = client.post("/prever", json=CENARIO_EXEMPLO)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probabilidade_subida"] <= 1.0
    assert body["nivel_risco"] in {"baixo", "medio", "alto"}


@pytest.mark.skipif(not MODELO_EXISTE, reason="modelo ainda nao treinado - rode src/treino.py")
def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "melhor_modelo" in response.json()


def test_prever_rejeita_payload_invalido(client):
    payload_invalido = dict(CENARIO_EXEMPLO)
    payload_invalido["segmento"] = "segmento_que_nao_existe"
    response = client.post("/prever", json=payload_invalido)
    assert response.status_code == 422
