# 🌬️ Air Intel | Monitor Pro

Dashboard industrial de monitoreo y predicción de calidad de aire en tiempo real, diseñado para la plataforma **TechClass**.

## 🚀 Características Principales

- **Inteligencia Artificial Multiclase**: Clasificación en 3 niveles (*Good*, *Moderate*, *Poor*) mediante Redes Neuronales y Regresión Logística.
- **Auto-ML (Re-entrenamiento Continuo)**: El sistema detecta nuevos datos en el dataset y re-entrena los modelos automáticamente para mantener la precisión al máximo.
- **Visualización Pro**: Interfaz con diseño *Glassmorphism*, matrices de confusión dinámicas y feedback en tiempo real mediante notificaciones.
- **Análisis por Lotes**: Procesamiento masivo de datos mediante subida de archivos CSV con validación automática contra etiquetas reales.

## 🛠️ Tecnologías Usadas

- **Frontend**: Streamlit (Python)
- **Machine Learning**: Scikit-Learn (MLPClassifier, LogisticRegression)
- **Data Handling**: Pandas, Numpy
- **Optimización**: RandomizedSearchCV (Hiperparametrización de Red Neuronal)
- **Visualización**: Matplotlib

## 📦 Instalación Local

1. Clona este repositorio.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta la aplicación:
   ```bash
   streamlit run app.py
   ```

## 📊 Estructura del Proyecto

- `app.py`: Aplicación principal de Streamlit.
- `train_air_nn.py`: Script de entrenamiento de la Red Neuronal.
- `train_air_logistic.py`: Script de entrenamiento de Regresión Logística.
- `augmented_air_quality.csv`: Dataset principal con 650 muestras.
- `requirements.txt`: Dependencias del proyecto.

---
*Desarrollado con ❤️ para el análisis de datos industriales.*