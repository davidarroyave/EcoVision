#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
camera.py

Módulo que activa la camara en el computador integrando el modelo entrenado .pt
para el reconocimiento de botellas o latas visualizado el respectivo nivel de 
probabilidaden la interfaz gráfica.

"""
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Cargar modelo YOLO
model_path = 'best.pt'  # Asegúrate que esté en el mismo directorio o ruta correcta
model = YOLO(model_path)
confidence_threshold = 0.95

st.title("Detección en tiempo real con YOLO y Streamlit")

# Cámara desde navegador (Streamlit captura foto/video)
camera_input = st.camera_input("Activa tu cámara")

if camera_input:
    # Leer imagen desde cámara (bytes -> OpenCV)
    img = Image.open(camera_input)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Predecir con modelo YOLO sin bloquear la app
    results = model.predict(source=frame, conf=confidence_threshold)
    result = results[0]
    names = result.names
    frame_display = frame.copy()
    detected_lata = False

    # Dibujar detecciones
    for box in result.boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, class_id = box
        if score > confidence_threshold:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            class_name = names[int(class_id)]

            if class_name == 'lata':
                detected_lata = True
                color = (0, 255, 0)  # verde
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 3)
                label = f"{class_name.upper()} {score*100:.1f}%"
                cv2.putText(frame_display, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Convertir imagen para mostrar en Streamlit
    frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
    st.image(frame_display, caption="Detección en tiempo real", use_column_width=True)
else:
    st.write("Por favor, activa la cámara para comenzar la detección.")
