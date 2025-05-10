# Diagnóstico Médico Automatizado con Inteligencia Artificial

Este proyecto presenta un prototipo de sistema de diagnóstico médico automatizado desarrollado con Python, implementado mediante el algoritmo de aprendizaje supervisado **k-Nearest Neighbors (k-NN)**. Está diseñado para predecir enfermedades comunes a partir de síntomas ingresados por el usuario, funcionando como una herramienta de apoyo en entornos educativos o de salud digital.

## 🧠 Descripción del Proyecto

El sistema solicita al usuario ingresar nueve síntomas clave y, con base en esos datos, predice la enfermedad más probable entre un conjunto de **30 enfermedades** modeladas con vectores binarios. La implementación se realiza utilizando la biblioteca **Scikit-learn**, y puede ejecutarse en plataformas como **Google Colab** o entornos locales de Python.

## ⚙️ Tecnologías Utilizadas

- **Python 3.10+**
- **Scikit-learn**
- **Google Colab / Jupyter Notebook**
- **Algoritmo: k-Nearest Neighbors (k=3)**

## 📁 Estructura del Proyecto

```
Diagnostico_AI/
│
├── diagnostico_ai.py           # Código principal del prototipo
├── sintomas_dataset.csv        # Dataset con 30 enfermedades y 9 síntomas (opcional)
├── README.md                   # Descripción del proyecto
├── informe_tecnico.docx        # Documento académico complementario
```

## 🚀 Instrucciones de Uso

1. Clona este repositorio o descarga los archivos.
2. Asegúrate de tener Python y Scikit-learn instalados:

   ```bash
   pip install scikit-learn
   ```

3. Ejecuta el script en tu entorno preferido (Google Colab, Jupyter Notebook, VSCode, etc.):

   ```bash
   python diagnostico_ai.py
   ```

4. Ingresa los síntomas solicitados (1 = Sí, 0 = No) cuando se te indique por consola.

5. El sistema imprimirá el diagnóstico sugerido basado en los síntomas ingresados.

## 📊 Dataset

El sistema se basa en un conjunto de datos estructurado con 30 enfermedades comunes y 9 síntomas. Cada enfermedad está representada por un vector binario indicando la presencia o ausencia de los siguientes síntomas:

- Tos
- Fiebre
- Dolor de cabeza
- Fatiga
- Dolor muscular
- Dolor de garganta
- Congestión nasal
- Dificultad para respirar
- Pérdida de apetito

## ✅ Resultados Esperados

El modelo alcanza una **precisión aceptable** en la predicción, siendo adecuado como herramienta educativa o prototipo funcional para diagnóstico preliminar.

## 🧪 Mejoras Futuras

- Incorporación de variables clínicas adicionales (edad, sexo, historial médico).
- Implementación de algoritmos alternativos como **Random Forest** o **Redes Neuronales**.
- Desarrollo de una **interfaz gráfica** (GUI) con Tkinter o Streamlit.
- Integración de un **módulo de recomendación médica** basado en el diagnóstico.

## 👨‍⚕️ Autores

Pablo Emilio Escobar Ossa  
Cristian Xavier Sanchez Valencia

Proyecto presentado al director **Daniel Andrés Guzmán**  
Curso: **Proyecto de Grado – 202016907_56**  
Universidad Nacional Abierta y a Distancia (UNAD)  
**Mayo de 2025**
