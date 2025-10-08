# src/data/register.py
import argparse
import mlflow
import mlflow.pytorch

def parse_args():
    parser = argparse.ArgumentParser(description="Registrar modelo en MLflow")
    parser.add_argument("--run_id", type=str, required=True, help="ID del run que deseas registrar")
    parser.add_argument("--model_name", type=str, default="EcoVisionModel", help="Nombre del modelo en MLflow Registry")
    parser.add_argument("--artifact_path", type=str, default="model", help="Ruta del artefacto dentro del run")
    return parser.parse_args()

def main():
    args = parse_args()
    mlflow.set_tracking_uri("file:./mlruns")  # Tracking local

    # Registrar el modelo desde un run existente
    model_uri = f"runs:/{args.run_id}/{args.artifact_path}"
    registered_model = mlflow.register_model(model_uri=model_uri, name=args.model_name)

    print(f"✅ Modelo registrado exitosamente: {registered_model.name}, versión {registered_model.version}")

if __name__ == "__main__":
    main()
