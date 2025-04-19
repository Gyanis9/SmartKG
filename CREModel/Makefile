# Makefile
start:
	@echo "启动开发服务..."
	PYTHONPATH=. python -m app.main

test:
	@pytest -v tests/

lint:
	@flake8 app/ tests/
	@mypy app/

docker-build:
	@docker build -t kg-model-service:v1.0 .

docker-run:
	@docker run -p 50051:50051 -p 8000:8000 kg-model-service:v1.0