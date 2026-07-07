"""
API que serve o modelo de risco de inadimplencia de credito, treinado com
dados reais do Banco Central do Brasil.

Rodar localmente:
    uvicorn api.main:app --reload --port 8000

Documentacao interativa:
    http://localhost:8000/docs
"""
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.processamento import build_feature_frame  # noqa: E402
from api.schemas import CenarioMacro, InfoModeloResponse, PrevisaoResponse  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_modelo = None
_colunas = None
_metricas = None


def carregar_artefatos():
    """Carrega o modelo treinado e metadados do disco para as variaveis
    globais acima. Chamado pelo lifespan da aplicacao (producao) e
    tambem pelos testes (via `with TestClient(app)`)."""
    global _modelo, _colunas, _metricas
    caminho_modelo = MODELS_DIR / "modelo_risco_credito.pkl"
    caminho_features = MODELS_DIR / "feature_list.json"
    caminho_metricas = MODELS_DIR / "metrics.json"

    if not caminho_modelo.exists():
        return

    _modelo = joblib.load(caminho_modelo)
    _colunas = json.loads(caminho_features.read_text())
    if caminho_metricas.exists():
        _metricas = json.loads(caminho_metricas.read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    carregar_artefatos()
    yield


app = FastAPI(
    title="API de Risco de Inadimplencia de Credito (Brasil)",
    description=(
        "Preve se a taxa de inadimplencia de credito de um segmento "
        "(total / pessoa fisica / pessoa juridica) vai subir no proximo mes, "
        "com base em indicadores reais do Banco Central do Brasil (Selic, "
        "IPCA, desemprego, saldo de credito)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _nivel_risco(prob: float) -> str:
    if prob < 0.35:
        return "baixo"
    if prob < 0.65:
        return "medio"
    return "alto"


@app.get("/health")
def health():
    return {"status": "ok", "modelo_carregado": _modelo is not None}


@app.get("/model-info", response_model=InfoModeloResponse)
def model_info():
    if _metricas is None:
        raise HTTPException(status_code=503, detail="Metricas nao disponiveis. Rode src/treino.py primeiro.")
    return _metricas


@app.post("/prever", response_model=PrevisaoResponse)
def prever(cenario: CenarioMacro):
    if _modelo is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo nao carregado. Rode `python src/treino.py` para gerar models/modelo_risco_credito.pkl.",
        )

    linha = pd.DataFrame([cenario.model_dump()])
    X, _ = build_feature_frame(linha, fit_columns=_colunas)

    proba = float(_modelo.predict_proba(X)[0, 1])
    return PrevisaoResponse(
        probabilidade_subida=round(proba, 4),
        nivel_risco=_nivel_risco(proba),
        versao_modelo=(_metricas or {}).get("melhor_modelo", "desconhecido"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
