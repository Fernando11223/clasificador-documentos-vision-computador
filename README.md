# Clasificador Visual de Elementos Documentales (Visión por Computador)

Aplicación web modular desarrollada con **HTML5, CSS3, JavaScript y TensorFlow.js** para la clasificación automática de componentes estructurales en documentos (`Graficos`, `Tablas` y `Textos`), entrenada mediante *transfer learning* en **Teachable Machine**.

## Contexto Académico
Este prototipo corresponde a la asignatura de **Visión por Computador** (Maestría en Inteligencia Artificial) y constituye una línea base experimental vinculada al análisis de estructura documental (*layout analysis*) para la extracción y recuperación de información en documentos técnicos.

## Arquitectura del Proyecto
* `index.html`: Estructura semántica de la interfaz de carga, visualización por lote y panel de métricas.
* `styles.css`: Hoja de estilos con diseño modular y barras de probabilidad por clase.
* `classifier.js`: Carga del modelo de TensorFlow.js, inferencia sobre imágenes individuales o carpetas y gestión de eventos.
* `metrics.js`: Cálculo en tiempo real de Precisión, Recall, F1-Score, Matriz de Confusión y curvas ROC / AUC (One-vs-Rest).

## Modelo Entrenado
* **Plataforma:** Google Teachable Machine (Image Project)
* **Clases:** `Graficos`, `Tablas`, `Textos` (40 muestras por clase)
* **Endpoint TensorFlow.js:** `https://teachablemachine.withgoogle.com/models/gVhWkIiqS/`

## Uso
1. Abrir `index.html` en cualquier navegador moderno (o acceder mediante GitHub Pages).
2. Cargar una o varias imágenes de prueba, o seleccionar una carpeta organizada en subcarpetas (`Graficos`, `Tablas`, `Textos`).
3. Asignar o verificar la etiqueta real de cada muestra y presionar **Calcular Métricas y Curvas ROC** para visualizar la evaluación cuantitativa.