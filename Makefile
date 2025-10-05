# Makefile for EcoVision Project
# Sistema de detección de latas y botellas con TensorFlow y Streamlit
# Autores: Jose Luis Martinez Diaz, Juan David Arroyave Ramirez, Neiberth Aponte Aristizabal, Stevens Ricardo Bohorquez Ruiz
# Fecha: 2025-10

PYTHON = python3
PIP    = pip3
APP    = main.py
REQ    = requirements.txt
IMAGE  = ecovision:latest
PORT   = 8501
HOST   = 0.0.0.0

.PHONY: install run docker-build docker-run clean

install:
	$(PIP) install --upgrade pip
	@if [ -f $(REQ) ]; then \
		$(PIP) install -r $(REQ); \
		echo "Dependencias instaladas"; \
	else \
		echo "$(REQ) no encontrado"; exit 1; \
	fi

run:
	streamlit run $(APP) --server.port=$(PORT) --server.address=$(HOST)

docker-build:
	docker build -t $(IMAGE) .
	echo "Imagen $(IMAGE) construida"

docker-run:
	docker run -p $(PORT):$(PORT) --name ecovision-app $(IMAGE)

clean:
	rm -rf __pycache__ .pytest_cache
