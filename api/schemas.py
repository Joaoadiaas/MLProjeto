"""Modelos Pydantic de entrada/saida da API de risco de credito."""
from typing import Literal

from pydantic import BaseModel, Field


class CenarioMacro(BaseModel):
    """Indicadores macroeconomicos e o historico recente de inadimplencia
    de um segmento, usados para prever se a inadimplencia desse segmento
    vai subir no proximo mes."""

    segmento: Literal["total", "pf", "pj"] = Field(
        description="Segmento de credito: total (SFN), pf (pessoa fisica) ou pj (pessoa juridica)"
    )
    mes: int = Field(ge=1, le=12, description="Mes de referencia (1-12)")
    trimestre: int = Field(ge=1, le=4)
    tendencia: int = Field(ge=0, description="Numero de meses desde o inicio da serie historica usada no treino")
    selic_mensal: float = Field(description="Taxa Selic acumulada no mes (%)")
    selic_acum_3m: float = Field(description="Selic acumulada nos ultimos 3 meses (%)")
    ipca_mensal: float = Field(description="Variacao mensal do IPCA (%)")
    ipca_acum_12m: float = Field(description="IPCA acumulado nos ultimos 12 meses (%)")
    desemprego: float = Field(description="Taxa de desocupacao PNAD Continua (%)")
    saldo_credito_var_mensal: float = Field(description="Variacao % do saldo da carteira de credito no mes")
    saldo_credito_var_12m: float = Field(description="Variacao % do saldo da carteira de credito em 12 meses")
    inadimplencia_lag1: float = Field(description="Taxa de inadimplencia do segmento no mes anterior (%)")
    inadimplencia_lag2: float = Field(description="Taxa de inadimplencia do segmento ha 2 meses (%)")
    inadimplencia_lag3: float = Field(description="Taxa de inadimplencia do segmento ha 3 meses (%)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "segmento": "pf",
                "mes": 5,
                "trimestre": 2,
                "tendencia": 170,
                "selic_mensal": 1.07,
                "selic_acum_3m": 3.29,
                "ipca_mensal": 0.58,
                "ipca_acum_12m": 5.2,
                "desemprego": 5.6,
                "saldo_credito_var_mensal": 0.3,
                "saldo_credito_var_12m": 6.9,
                "inadimplencia_lag1": 7.42,
                "inadimplencia_lag2": 7.17,
                "inadimplencia_lag3": 7.06,
            }
        }
    }


class PrevisaoResponse(BaseModel):
    probabilidade_subida: float
    nivel_risco: Literal["baixo", "medio", "alto"]
    versao_modelo: str


class InfoModeloResponse(BaseModel):
    melhor_modelo: str
    candidatos: dict
    periodo_treino: list
    periodo_teste: list
