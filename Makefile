.PHONY: run docker-build docker-up docker-down docker-logs lint format clean

run:
	streamlit run app.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

lint:
	ruff check .

format:
	black .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf uploads/
