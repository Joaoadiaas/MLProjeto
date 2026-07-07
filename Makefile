.PHONY: setup data train test api dashboard monitor docker-build docker-up clean

setup:
	pip install -r requirements.txt

data:
	python src/generate_data.py

train:
	python src/train.py

test:
	pytest -v

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run app/dashboard.py

monitor:
	python src/monitoring.py

docker-build:
	docker compose build

docker-up:
	docker compose up

clean:
	rm -rf models/*.pkl models/*.json models/plots data/processed/*.csv
