"""
treino.py
---------
Treina e compara tres classificadores para prever se a inadimplencia de
credito vai SUBIR no proximo mes (por segmento: total / PF / PJ), usando
dados reais do Banco Central. Seleciona o melhor por ROC-AUC no periodo de
teste (mais recente, nunca visto no treino) e salva:

    models/modelo_risco_credito.pkl  - pipeline vencedor (joblib)
    models/feature_list.json         - colunas esperadas na inferencia
    models/metrics.json              - metricas de cada modelo candidato
    models/plots/*.png                - curva ROC, matriz de confusao, importancia

Rodar:
    python src/treino.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from processamento import carregar_e_dividir, salvar_processados

MODELS_DIR = Path("models")
PLOTS_DIR = MODELS_DIR / "plots"


def obter_candidatos():
    return {
        "regressao_logistica": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=5, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42,
            )),
        ]),
    }


def avaliar(modelo, X_teste, y_teste):
    proba = modelo.predict_proba(X_teste)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_teste, proba), 4),
        "acuracia": round(accuracy_score(y_teste, pred), 4),
        "precisao": round(precision_score(y_teste, pred), 4),
        "recall": round(recall_score(y_teste, pred), 4),
        "f1": round(f1_score(y_teste, pred), 4),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X_treino, X_teste, y_treino, y_teste, colunas, treino_df, teste_df = carregar_e_dividir()
    salvar_processados(X_treino, X_teste, y_treino, y_teste, colunas)

    candidatos = obter_candidatos()
    resultados = {}
    ajustados = {}

    for nome, pipe in candidatos.items():
        pipe.fit(X_treino, y_treino)
        metricas = avaliar(pipe, X_teste, y_teste)
        resultados[nome] = metricas
        ajustados[nome] = pipe
        print(f"{nome:22s} -> {metricas}")

    melhor_nome = max(resultados, key=lambda k: resultados[k]["roc_auc"])
    melhor_modelo = ajustados[melhor_nome]
    print(f"\nMelhor modelo: {melhor_nome} (ROC-AUC={resultados[melhor_nome]['roc_auc']})")

    joblib.dump(melhor_modelo, MODELS_DIR / "modelo_risco_credito.pkl")
    (MODELS_DIR / "feature_list.json").write_text(json.dumps(colunas, indent=2))
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps({
            "melhor_modelo": melhor_nome,
            "candidatos": resultados,
            "periodo_treino": [str(treino_df["data"].min().date()), str(treino_df["data"].max().date())],
            "periodo_teste": [str(teste_df["data"].min().date()), str(teste_df["data"].max().date())],
        }, indent=2, ensure_ascii=False)
    )

    # ---- graficos ----
    proba = melhor_modelo.predict_proba(X_teste)[:, 1]
    RocCurveDisplay.from_predictions(y_teste, proba)
    plt.title(f"Curva ROC - {melhor_nome}")
    plt.savefig(PLOTS_DIR / "curva_roc.png", bbox_inches="tight", dpi=120)
    plt.close()

    pred = (proba >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_teste, pred, display_labels=["Estavel/queda", "Sobe"]
    )
    plt.title(f"Matriz de confusao - {melhor_nome}")
    plt.savefig(PLOTS_DIR / "matriz_confusao.png", bbox_inches="tight", dpi=120)
    plt.close()

    clf = melhor_modelo.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importancias = pd.Series(clf.feature_importances_, index=colunas)
        importancias.nlargest(15).sort_values().plot(kind="barh", figsize=(7, 6))
        plt.title(f"Top 15 features mais importantes - {melhor_nome}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "importancia_features.png", dpi=120)
        plt.close()
    elif hasattr(clf, "coef_"):
        importancias = pd.Series(clf.coef_[0], index=colunas)
        importancias.reindex(importancias.abs().nlargest(15).index).sort_values().plot(
            kind="barh", figsize=(7, 6)
        )
        plt.title(f"Top 15 coeficientes (|peso|) - {melhor_nome}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "importancia_features.png", dpi=120)
        plt.close()

    print(f"\nArtefatos salvos em {MODELS_DIR}/")


if __name__ == "__main__":
    main()
