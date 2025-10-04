# 🐳 Docker Setup - EcoVision

> EcoVision: Sistema de inteligencia artificial basado en visión por computadora para la detección y segmentación de latas y botellas, desplegado con Streamlit.

## 📋 Información General

| **Campo** | **Valor** |
|-----------|-----------|
| **🏷️ Repositorio** | `davidjonesja/ecovision` |
| ** Tamaño** | 3.89GB |
| ** Python** | 3.11.13 |
| ** TensorFlow** | 2.20.0 |
| ** OpenCV** | 4.11.0 |
| ** Última actualización** | 5 de Octubre, 2025 |

## 🚀 Inicio Rápido

### Opción 1: Usando imagen pre-construida (Recomendado)

```bash
# 1. Descargar imagen
docker pull davidjonesja/ecovision:latest

# 2. Ejecutar aplicación (Linux/macOS)
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --name ecovision \
  davidjonesja/ecovision:latest
```

### Opción 2: Construir desde código fuente

```bash
# 1. Clonar repositorio
git clone https://github.com/davidarroyave/ecovision
cd ecovision

# 2. Construir imagen
docker build -t ecovision:latest .

# 3. Ejecutar contenedor
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/data:/app/data \
  --name ecovision \
  ecovision:latest
```

## 🖥️ Configuración por Sistema Operativo

### 🐧 Linux

```bash
# Permitir conexiones X11
xhost +local:docker

# Ejecutar con GUI
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/data:/app/data \
  --name ecovision \
  ecovision:latest
```

### 🍎 macOS

```bash
# 1. Instalar XQuartz (si no lo tienes)
brew install --cask xquartz

# 2. Configurar display
xhost +localhost

# 3. Ejecutar contenedor
docker run -it --rm \
  -e DISPLAY=host.docker.internal:0 \
  -v $(pwd)/data:/app/data \
  --name ecovision \
  ecovision:latest
```

### 🪟 Windows (WSL2 + Xming)

#### Paso 1: Instalar y configurar Xming

1. **Descargar Xming**: https://sourceforge.net/projects/xming/
2. **Durante la instalación**: ☑️ Marcar "Disable access control"
3. **Ejecutar Xming**: `Xming :0 -multiwindow -clipboard -ac`

#### Paso 2: Ejecutar contenedor

```bash
# Detectar IP de Windows automáticamente
WINDOWS_IP=$(ip route show | grep -i default | awk '{ print $3}')
export DISPLAY="$WINDOWS_IP:0.0"

# Ejecutar con GUI
docker run --rm \
    -e DISPLAY="$DISPLAY" \
    -e GDK_BACKEND=x11 \
    -e XDG_SESSION_TYPE=x11 \
    -v $(pwd)/data:/app/data \
    --net=host \
    ecovision:latest
```

#### Script automatizado para Windows

```bash
#!/bin/bash
# Crear archivo: run_docker_windows.sh

WINDOWS_IP=$(ip route show | grep -i default | awk '{ print $3}')
export DISPLAY="$WINDOWS_IP:0.0"

echo "🚀 Iniciando UAO ECOVISION con GUI..."
echo "📡 Windows IP: $WINDOWS_IP"
echo "🖥️ Display: $DISPLAY"

docker run --rm \
    -e DISPLAY="$DISPLAY" \
    -e GDK_BACKEND=x11 \
    -e XDG_SESSION_TYPE=x11 \
    -e QT_QPA_PLATFORM=xcb \
    -v $(pwd)/data:/app/data \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --net=host \
    ecovision:latest
```

```bash
# Hacer ejecutable y correr
chmod +x run_docker_windows.sh
./run_docker_windows.sh
```

## 🔧 Comandos Adicionales

### Validación rápida (sin GUI)

```bash
# Test básico del sistema
docker run --rm ecovision:latest python -c "
from src.data.integrator import predict;
print('✅ Sistema ECOVISION funcionando correctamente')
"

# Verificar versiones
docker run --rm ecovision:latest python -c "
import tensorflow as tf;
import cv2;
print(f'TensorFlow: {tf.__version__}');
print(f'OpenCV: {cv2.__version__}')
"
```

### Modo desarrollo/debugging

```bash
# Ejecutar en modo interactivo
docker run --rm -it \
  -v $(pwd):/workspace \
  ecovision:latest bash

# Ejecutar con volúmenes de desarrollo
docker run --rm -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/data:/app/data \
  ecovision:latest
```

## 🔍 Funcionalidades de la Aplicación

Una vez que la aplicación se ejecute con GUI, podrás:

- **📁 Cargar imágenes**: Soporte para PNG, JPG, MP4, AVI
- **🔍 Detectar objetos**: Identificación automática de latas y botellas
- **✂️ Segmentar objetos**: Máscaras precisas de segmentación
- **🎨 Visualizar resultados**: Bounding boxes y máscaras de segmentación
- **📊 Análisis en tiempo real**: Procesamiento de video en vivo
- **💾 Guardar resultados**: Imágenes procesadas y métricas de detección

## 🐛 Troubleshooting

### ❌ GUI no aparece (Linux/macOS)

```bash
# Verificar X11 forwarding
echo $DISPLAY

# Permitir conexiones X11
xhost +local:docker

# Test básico X11
docker run --rm -e DISPLAY=$DISPLAY --net=host ecovision:latest python -c "
import tkinter as tk
root = tk.Tk()
root.title('Test UAO')
root.geometry('300x200')
root.mainloop()
"
```

### ❌ GUI no aparece (Windows)

```bash
# 1. Verificar que Xming esté ejecutándose
netstat -an | find "6000"
# Debe mostrar: TCP 0.0.0.0:6000 0.0.0.0:0 LISTENING

# 2. Verificar IP de Windows
ip route show | grep default

# 3. Configurar DISPLAY manualmente
export DISPLAY="172.27.208.1:0.0"  # Usar tu IP

# 4. Reiniciar Xming con configuración correcta
# Xming :0 -multiwindow -clipboard -ac
```

### ❌ Errores de permisos

```bash
# Linux/macOS
xhost +local:root

# Windows WSL2
xhost +local:docker
```

### ❌ Espacio insuficiente

```bash
# Limpiar contenedores no utilizados
docker system prune -a

# Ver uso de espacio
docker system df
```

## 📊 Especificaciones Técnicas

### Stack Tecnológico
- **🐍 Python**: 3.11.13
- **🤖 Machine Learning**: TensorFlow 2.20.0
- **📷 Visión Computacional**: OpenCV 4.11.0
- **🧮 Computación**: NumPy 1.26.4
- **🖼️ GUI**: Tkinter + X11 forwarding
- **📦 Gestión de paquetes**: UV Package Manager

### Arquitectura del Contenedor
- **Multi-stage build** para optimización de tamaño
- **Usuario no-root** para seguridad
- **Variables de entorno** optimizadas para Python
- **Volúmenes montados** para persistencia de datos

### Métricas de Performance
- **Tamaño de imagen**: 3.89GB
- **Tiempo de build**: ~15 minutos
- **Tiempo de startup**: ~10 segundos
- **Uso de RAM**: ~2GB (con TensorFlow cargado)
- **Optimizado para CPU**: Instrucciones AVX2, FMA, oneDNN

## 🔐 Consideraciones de Seguridad

- ✅ **Usuario no-root**: El contenedor ejecuta con usuario `app`
- ✅ **Dependencias mínimas**: Solo las necesarias para runtime
- ✅ **Archivos sensibles excluidos**: `.dockerignore` configurado
- ✅ **Variables de entorno**: No secretos hardcodeados

## 🆘 Soporte y Documentación

### Enlaces útiles
- **Docker Hub**: https://hub.docker.com/r/davidjonesja/ecovision
- **Repositorio**: https://github.com/davidarroyave/ecovision
- **Documentación Xming**: Incluida en `Xming_Setup_Guide.md`
- **Reporte de validación**: Ver `Docker_Validation_Report.md`

### Comandos de diagnóstico

```bash
# Estado de Docker
docker --version
docker info

# Imágenes disponibles
docker images ecovision

# Contenedores ejecutándose
docker ps

# Logs del contenedor
docker logs <container_id>
```

## 🎯 Estado del Proyecto

### ✅ Completamente validado
- **Dependencias**: Todas funcionando correctamente
- **GUI**: Compatible con Linux, macOS y Windows
- **Machine Learning**: TensorFlow y OpenCV operativos
- **Seguridad**: Usuario no-root implementado
- **Performance**: Optimizado para producción

### 🚀 Listo para usar
El contenedor Docker está completamente funcional y listo para uso en entornos de desarrollo, investigación y producción médica.

---

*Documentación generada el 5 de Octubre, 2025*  
*ECOVISION - Versión Docker*