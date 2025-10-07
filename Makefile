# Makefile for EcoVision Project
# Sistema de detección de latas y botellas con TensorFlow y Streamlit
# Autores: Jose Luis Martinez Diaz, Juan David Arroyave Ramirez, Neiberth Aponte Aristizabal, Stevens Ricardo Bohorquez Ruiz
# Fecha: 2025-10

MODELO = models/yolov8s-seg.pt
MODEL = src/data/models/
DATA = datasets/20251007v1/data.yaml
EPOCS = 1
IMGSZ = 640
RUN_NAME = EcoVision_Run
APP    = main.py
REQ    = requirements.txt
IMAGE  = ecovision:latest
PORT   = 8501
MLFLOW_PORT = 5000

all: train

# Entrenar modelo
train:
	python src/train.py --model $(MODELO) --data $(DATA) --epochs $(EPOCHS) --imgsz $(IMGSZ) --run_name "$(RUN_NAME)"

# Abrir MLflow UI
mlflow:
	mlflow ui --backend-store-uri file:./mlruns --port $(MLFLOW_PORT)

	
# Limpiar artefactos de entrenamiento
cleanml:
	rm -rf runs mlruns

register:
	@echo "Registrando modelo en MLflow..."
	@python src/data/register_model.py

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
