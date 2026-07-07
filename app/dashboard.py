"""
Dashboard Streamlit - Monitor de Risco de Inadimplencia de Credito (Brasil)

Mostra o historico real de inadimplencia (Banco Central), permite simular
um cenario macroeconomico e ver a previsao do modelo, e traz a
explicabilidade (SHAP) de cada previsao.

Rodar:
    streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.processamento import build_feature_frame, montar_painel_wide  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

st.set_page_config(page_title="Risco de Inadimplência - Brasil", layout="wide")


@st.cache_resource
def carregar_artefatos():
    modelo = joblib.load(MODELS_DIR / "modelo_risco_credito.pkl")
    colunas = json.loads((MODELS_DIR / "feature_list.json").read_text())
    metricas = json.loads((MODELS_DIR / "metrics.json").read_text())
    return modelo, colunas, metricas


@st.cache_data
def carregar_painel_historico():
    return montar_painel_wide()


st.title("Monitor de Risco de Inadimplência de Crédito - Brasil")
st.caption(
    "Dados reais do Banco Central do Brasil (SGS): inadimplência da carteira "
    "de crédito, Selic, IPCA, desemprego e saldo de crédito."
)

if not (MODELS_DIR / "modelo_risco_credito.pkl").exists():
    st.warning("Nenhum modelo treinado ainda. Rode `python src/treino.py` e recarregue esta página.")
    st.stop()

modelo, colunas, metricas = carregar_artefatos()
painel = carregar_painel_historico()

with st.sidebar:
    st.header("Sobre o modelo")
    st.metric("Melhor modelo", metricas["melhor_modelo"])
    st.metric("ROC-AUC (teste)", metricas["candidatos"][metricas["melhor_modelo"]]["roc_auc"])
    st.caption(f"Treino: {metricas['periodo_treino'][0]} a {metricas['periodo_treino'][1]}")
    st.caption(f"Teste: {metricas['periodo_teste'][0]} a {metricas['periodo_teste'][1]}")

aba_historico, aba_simulador = st.tabs(["Histórico real (BCB)", "Simular cenário"])

with aba_historico:
    st.subheader("Taxa de inadimplência por segmento (dados reais)")
    st.line_chart(
        painel.set_index("data")[["inad_total", "inad_pf", "inad_pj"]].rename(
            columns={"inad_total": "Total", "inad_pf": "Pessoa Física", "inad_pj": "Pessoa Jurídica"}
        )
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selic mensal (%)")
        st.line_chart(painel.set_index("data")[["selic_mensal"]])
    with col2:
        st.subheader("Desemprego (%)")
        st.line_chart(painel.set_index("data")[["desemprego"]])

    st.dataframe(
        painel.tail(12)[["data", "inad_total", "inad_pf", "inad_pj", "selic_mensal", "desemprego"]],
        use_container_width=True,
    )

with aba_simulador:
    st.subheader("Simular um cenário macroeconômico")
    ultimo = painel.iloc[-1]

    col1, col2, col3 = st.columns(3)
    with col1:
        segmento = st.selectbox("Segmento", ["total", "pf", "pj"])
        mes = st.slider("Mês", 1, 12, int(ultimo["data"].month))
        trimestre = (mes - 1) // 3 + 1
        tendencia = st.number_input("Tendência (meses desde início da série)", value=len(painel), step=1)

    with col2:
        selic_mensal = st.number_input("Selic mensal (%)", value=float(ultimo["selic_mensal"]))
        selic_acum_3m = st.number_input("Selic acumulada 3m (%)", value=float(selic_mensal * 3))
        ipca_mensal = st.number_input("IPCA mensal (%)", value=float(ultimo["ipca_mensal"]))
        ipca_acum_12m = st.number_input("IPCA acumulado 12m (%)", value=5.0)

    with col3:
        desemprego = st.number_input("Desemprego (%)", value=float(ultimo["desemprego"]))
        saldo_var_m = st.number_input("Variação mensal saldo crédito (%)", value=0.5)
        saldo_var_12m = st.number_input("Variação 12m saldo crédito (%)", value=7.0)

    col4, col5, col6 = st.columns(3)
    col_map = {"total": "inad_total", "pf": "inad_pf", "pj": "inad_pj"}
    valor_atual = float(ultimo[col_map[segmento]])
    with col4:
        lag1 = st.number_input("Inadimplência mês anterior (%)", value=valor_atual)
    with col5:
        lag2 = st.number_input("Inadimplência há 2 meses (%)", value=valor_atual)
    with col6:
        lag3 = st.number_input("Inadimplência há 3 meses (%)", value=valor_atual)

    cenario = pd.DataFrame([{
        "segmento": segmento, "mes": mes, "trimestre": trimestre, "tendencia": tendencia,
        "selic_mensal": selic_mensal, "selic_acum_3m": selic_acum_3m,
        "ipca_mensal": ipca_mensal, "ipca_acum_12m": ipca_acum_12m,
        "desemprego": desemprego, "saldo_credito_var_mensal": saldo_var_m,
        "saldo_credito_var_12m": saldo_var_12m,
        "inadimplencia_lag1": lag1, "inadimplencia_lag2": lag2, "inadimplencia_lag3": lag3,
    }])

    if st.button("Prever risco", type="primary"):
        X, _ = build_feature_frame(cenario, fit_columns=colunas)
        proba = float(modelo.predict_proba(X)[0, 1])

        if proba < 0.35:
            risco = "Baixo"
        elif proba < 0.65:
            risco = "Médio"
        else:
            risco = "Alto"

        st.metric("Probabilidade de a inadimplência subir no próximo mês", f"{proba:.1%}")
        st.markdown(f"### Nível de risco: {risco}")

        st.subheader("Por quê? (explicação via SHAP)")
        clf = modelo.named_steps["clf"]
        explicacao_ok = True
        try:
            if hasattr(clf, "feature_importances_"):
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X)
                sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            else:
                X_escalado = modelo.named_steps["scaler"].transform(X)
                explainer = shap.LinearExplainer(clf, X_escalado)
                sv = explainer.shap_values(X_escalado)[0]
        except Exception as e:  # pragma: no cover - explicabilidade e best-effort
            explicacao_ok = False
            st.info(f"Explicação SHAP indisponível para este tipo de modelo: {e}")

        if explicacao_ok:
            contrib = pd.Series(sv, index=colunas).sort_values(key=abs, ascending=False).head(10)
            st.bar_chart(contrib)
            st.caption("Top 10 features por |valor SHAP| - o que mais empurrou a previsão para cima ou para baixo.")
