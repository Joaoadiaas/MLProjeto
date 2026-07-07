.PHONY: setup dados treino teste api dashboard monitorar docker-build docker-up limpar

setup:
	pip install -r requirements.txt

dados:
	python src/coleta_dados.py

treino:
	python src/treino.py

teste:
	pytest -v

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run app/dashboard.py

monitorar:
	python src/monitoramento.py

docker-build:
	docker compose build

docker-up:
	docker compose up

limpar:
	rm -rf models/*.pkl models/*.json models/plots models