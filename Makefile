install:
	poetry install

train:
	poetry run python run_pipeline.py

api:
	poetry run uvicorn api.main:app --reload