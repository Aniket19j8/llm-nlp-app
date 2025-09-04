.PHONY: run api ui test docker build push

run: api

api:
	uvicorn app.main:app --reload --port 8000

ui:
	BACKEND_URL=http://localhost:8000 streamlit run ui/streamlit_app.py --server.port 8501

test:
	pytest -q

docker:
	docker compose up --build

build:
	docker build -t your_dockerhub_username/llm-nlp-app:latest .

push:
	docker push your_dockerhub_username/llm-nlp-app:latest
