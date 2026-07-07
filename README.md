# Monitor de Risco de Inadimplência de Crédito — Brasil

Sistema de machine learning que prevê se a taxa de inadimplência de crédito
no Brasil (total, pessoa física ou pessoa jurídica) vai **subir no próximo
mês**, usando dados reais e públicos do Banco Central do Brasil.

Construído para mostrar o ciclo completo que uma vaga de Dados/ML espera:
entendimento de um problema real, modelagem, avaliação honesta, entrega como
serviço (API + dashboard), testes automatizados e monitoramento de produção —
não só um notebook com uma métrica.

## Por que esse problema

Times de risco de crédito de bancos e fintechs brasileiras acompanham de
perto a trajetória da inadimplência para ajustar política de concessão de
crédito, apetite de risco e provisão para devedores duvidosos (PDD). Este
projeto simula esse tipo de sinal de alerta antecipado: dado o cenário
macroeconômico atual (Selic, inflação, desemprego, volume de crédito) e o
histórico recente de inadimplência, o modelo estima a probabilidade de a
inadimplência **piorar** no mês seguinte.

Diferente da maioria dos projetos de portfólio, aqui os dados **não são
sintéticos**: vêm da API pública do Banco Central (SGS), a mesma fonte usada
por analistas de mercado e áreas de risco.

## Fonte dos dados (100% real e pública)

API do Banco Central: [`api.bcb.gov.br`](https://api.bcb.gov.br) — Sistema
Gerenciador de Séries Temporais (SGS).

| Série | Código SGS | Descrição |
|---|---|---|
| Inadimplência total | 21082 | % da carteira de crédito do SFN com parcela vencida há mais de 90 dias |
| Inadimplência PF | 21112 | Idem, recursos livres — pessoa física |
| Inadimplência PJ | 21086 | Idem, recursos livres — pessoa jurídica |
| Selic mensal | 4390 | Taxa Selic acumulada no mês |
| IPCA mensal | 433 | Variação mensal do índice de inflação oficial |
| Desemprego | 24369 | Taxa de desocupação (PNAD Contínua/IBGE) |
| Saldo de crédito | 20542 | Saldo da carteira de crédito com recursos livres (R$ milhões) |

Histórico: **março/2011 a maio/2026** (~15 anos). `src/coleta_dados.py`
baixa os dados mais recentes diretamente da API; uma cópia usada para
construir este projeto fica salva em `data/raw/*.json` para o pipeline
funcionar mesmo offline.

## Arquitetura

```
   API do Banco Central (SGS)
              │
      src/coleta_dados.py        baixa/atualiza as series reais
              │
      src/processamento.py       junta as series num painel mensal,
              │                  cria lags, features macro e o alvo
              │                  "a inadimplencia sobe no mes seguinte?"
              │                  split CRONOLOGICO (treino=passado, teste=futuro)
              │
        src/treino.py            treina e compara 3 modelos,
              │                  seleciona o melhor por ROC-AUC
              │
   ┌──────────┴──────────┐
   │                      │
models/modelo_risco_       models/metrics.json
credito.pkl                       │
   │                              │
   ├─────────────┬────────────────┘
   │             │
api/main.py   src/monitoramento.py
POST /prever   PSI: detecta mudanca de regime
   │           economico entre treino e producao
app/dashboard.py
historico real + simulador de cenario + explicabilidade SHAP
```

## Estrutura do projeto

```
├── data/raw/bcb_*.json          # series reais baixadas do Banco Central
├── notebooks/01_eda.ipynb       # analise exploratoria dos dados reais
├── src/
│   ├── coleta_dados.py          # baixa/atualiza as series do BCB
│   ├── processamento.py         # painel mensal, features, alvo, split cronologico
│   ├── treino.py                 # treina + compara modelos, salva o melhor
│   └── monitoramento.py          # deteccao de drift via PSI
├── api/
│   ├── main.py                   # API FastAPI (/prever, /health, /model-info)
│   └── schemas.py                 # validacao de entrada/saida
├── app/dashboard.py               # Streamlit: historico real + simulador + SHAP
├── tests/                         # suite pytest
├── models/                        # artefatos treinados (gerados, gitignored)
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml       # testes + lint a cada push
```

## Como rodar

```bash
git clone <este-repositorio>
cd <pasta-do-projeto>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make setup            # pip install -r requirements.txt

make dados            # baixa os dados mais recentes do Banco Central
make treino           # treina 3 modelos, salva o melhor em models/
make teste            # roda a suite pytest

make api              # http://localhost:8000/docs
make dashboard        # http://localhost:8501
```

Ou com Docker:

```bash
docker compose up --build
```

## Abordagem de modelagem

`src/treino.py` treina e compara três classificadores num split
**cronológico** (nunca aleatório — treinar com dados do futuro para prever o
passado seria vazamento de informação):

| Modelo | Por que está aqui |
|---|---|
| Regressão Logística (padronizada, `class_weight="balanced"`) | Baseline interpretável — coeficientes explicáveis a uma área de negócio/risco |
| Random Forest | Captura interações não-lineares (ex: Selic alta + desemprego subindo) sem cruzar features manualmente |
| Gradient Boosting | Geralmente o mais forte em dados tabulares; benchmark para os outros dois |

O painel de dados é montado em formato longo (uma linha por mês × segmento
de crédito), o que triplica o número de amostras em relação a usar só a
série total — importante porque séries macro mensais geram poucas
observações por natureza (~170 meses de histórico).

Rode `python src/treino.py` para gerar/atualizar os números reais desta
tabela na sua máquina:

```
$ cat models/metrics.json
```

## API

```bash
curl -X POST http://localhost:8000/prever \
  -H "Content-Type: application/json" \
  -d '{
    "segmento": "pf", "mes": 5, "trimestre": 2, "tendencia": 170,
    "selic_mensal": 1.07, "selic_acum_3m": 3.29,
    "ipca_mensal": 0.58, "ipca_acum_12m": 5.2,
    "desemprego": 5.6,
    "saldo_credito_var_mensal": 0.3, "saldo_credito_var_12m": 6.9,
    "inadimplencia_lag1": 7.42, "inadimplencia_lag2": 7.17, "inadimplencia_lag3": 7.06
  }'
```

```json
{ "probabilidade_subida": 0.71, "nivel_risco": "alto", "versao_modelo": "random_forest" }
```

## Monitoramento

`src/monitoramento.py` calcula o **PSI (Population Stability Index)** por
feature entre o período de treino e o período mais recente — a técnica
padrão de mercado para detectar mudança silenciosa de cenário em produção.

```bash
python src/monitoramento.py
```

PSI < 0.1 → estável · 0.1–0.2 → drift moderado (observar) · > 0.2 → drift
significativo (investigar/retreinar).

**Achado real deste projeto:** ao comparar o período de treino
(2013–2023) com o período de teste mais recente (2023–2026), quase todas as
features macroeconômicas (Selic, IPCA acumulado, desemprego, saldo de
crédito) apresentaram drift significativo — o que reflete a mudança real de
regime econômico no Brasil nesses anos (juros e inflação em patamares bem
diferentes). Isso é exatamente o tipo de alerta que levaria uma área de
risco a reavaliar o modelo antes de confiar cegamente nele.

## Testes

```bash
pytest -v
```

- `test_processamento.py` — painel sem dados faltantes, split crono