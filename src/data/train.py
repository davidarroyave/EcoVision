# src/data/train.py
import argparse
from ultralytics import YOLO
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento YOLOv8 con MLflow tracking")
    parser.add_argument("--model", type=str, required=True, help="Ruta del modelo base YOLOv8 (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Ruta al archivo data.yaml del dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Número de épocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen")
    parser.add_argument("--run_name", type=str, default="EcoVision_Run", help="Nombre del run en MLflow")
    return parser.parse_args()

def main():
    args = parse_args()

    mlflow.set_tracking_uri("file:./mlruns")  # Tracking local
    mlflow.set_experiment("EcoVision")        # Elegir experimento
    with mlflow.start_run(run_name=args.run_name) as run:
        # Log parámetros
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("img_size", args.imgsz)
        mlflow.log_param("data", args.data)

        # Cargar modelo base
        model = YOLO(args.model)

        # Entrenamiento
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            name=args.run_name,
        )

        # --- Log de métricas de manera segura ---
        try:
            if hasattr(results.box, "loss") and len(results.box.loss) > 0:
                mlflow.log_metric("box_loss", float(results.box.loss[-1]))
            if hasattr(results.seg, "loss") and len(results.seg.loss) > 0:
                mlflow.log_metric("seg_loss", float(results.seg.loss[-1]))
            if hasattr(results.box, "map") and isinstance(results.box.map, dict) and 0.5 in results.box.map:
                mlflow.log_metric("mAP50", float(results.box.map[0.5]))
        except Exception as e:
            print(f"⚠️ No se pudieron loggear algunas métricas: {e}")

        # Guardar modelo entrenado como artefacto
        mlflow.pytorch.log_model(model.model, name="model")

        # --- Log del best.pt generado por YOLOv8 ---
        best_weights_path = os.path.join("runs", "segment", args.run_name, "weights", "best.pt")
        if os.path.exists(best_weights_path):
            mlflow.log_artifact(best_weights_path, artifact_path="model")
            print(f"✅ best.pt loggeado en MLflow bajo artifact_path='model'")
        else:
            print("⚠️ best.pt no encontrado, no se loggeo en MLflow")

        # --- Gráfico de pérdidas ---
        plt.figure(figsize=(8,6))
        if hasattr(results.box, "loss"):
            plt.plot(results.box.loss, label="box_loss")
        if hasattr(results.seg, "loss"):
            plt.plot(results.seg.loss, label="seg_loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training Loss")
        loss_plot_path = os.path.join("loss_plot.png")
        plt.savefig(loss_plot_path)
        mlflow.log_artifact(loss_plot_path)
        plt.close()

    print(f"✅ Entrenamiento finalizado. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
