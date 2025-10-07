# Makefile for EcoVision Project
# Sistema de detección de latas y botellas con TensorFlow y Streamlit
# Autores: Jose Luis Martinez Diaz, Juan David Arroyave Ramirez, Neiberth Aponte Aristizabal, Stevens Ricardo Bohorquez Ruiz
# Fecha: 2025-10

MODEL = src/data/models/
DATA =
EPOCS = 5
APP    = main.py
REQ    = requirements.txt
IMAGE  = ecovision:latest
PORT   = 8501
MLFLOW_PORT = 5000

all: train


#Streamlit
run:
	streamlit run main.py


#Docker
.PHONY: install run docker-build docker-run clean

docker-build:
	docker build -t $(IMAGE) .
	echo "Imagen $(IMAGE) construida"

docker-run:
	docker run -p $(PORT):$(PORT) --name ecovision-app $(IMAGE)

clean:
	rm -rf __pycache__ .pytest_cache
