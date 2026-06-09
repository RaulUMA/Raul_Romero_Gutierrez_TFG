# Raul_Romero_Gutierrez_TFG

TFG: Optimización de Precios Dinámicos en E-Commerce (Árboles de Decisión vs SVM)

**Autor:** Raúl Romero Gutiérrez  
**Titulación:** Grado en Ingeniería Informática (Mención en Computación)  
**Universidad:** Universidad de Málaga (UMA)  



Descripción del Proyecto

Este repositorio contiene el código fuente oficial desarrollado para mi Trabajo de Fin de Grado (TFG). El proyecto aborda la creación y evaluación de un sistema de recomendación de precios dinámicos para productos frescos y perecederos en el sector minorista. 

Se confrontan dos enfoques de Machine Learning: un modelo de "caja blanca" (**Árboles de Decisión**) centrado en la interpretabilidad comercial, y un modelo de "caja negra" (**Máquinas de Vector Soporte - SVM**) centrado en la complejidad matemática. El objetivo es determinar qué solución ofrece un mejor equilibrio entre el porcentaje de acierto y la transparencia para el negocio.



Estructura de Archivos

El proyecto está diseñado paso a paso. A continuación, se detalla qué hace exactamente cada archivo de forma sencilla:

* **`Muestras_aleatorias.py`**: Seleccionamos una parte del dataset para que el ordenador pueda analizarlos en un tiempo razonable.
* **`Variable_Objetivo.py`**: Convertimos los descuentos del supermercado en tres acciones: Mantener precio, Descuento Leve o Liquidación.
* **`Analisis_Exploratorio_de_datos.py`**: Buscamos qué relación existe entre variables del dataset.
* **`Arboles_de_decision.py`**: Enseñamos a la Inteligencia Artificial a recomendar precios utilizando el algoritmo "Árboles de Decisión".
* **`SVM.py`**: Enseñamos a la Inteligencia Artificial a recomendar precios utilizando el algoritmo "Support Vector Machine".
* **`Matrices_de_confusion.py`**: Creamos unos gráficos que nos permiten ver de forma clara cuántas veces acierta cada modelo y en qué tipo de situaciones se equivoca más.
* **`Importancia_variable.py`**: Generamos un gráfico que nos dice en qué se fija más la Inteligencia Artificial para cambiar un precio.
* **`Reglas_Arbol_de_decision.py`**: Extraemos las normas exactas que sigue el programa para cambiar un precio, escritas paso a paso para que cualquier trabajador pueda entenderlas.
* **`Analisis_Errores.py`**: Investigamos en qué situaciones concretas falla la Inteligencia Artificial.
* **`Visualizar_Arbol.py`**: Creamos un dibujo sencillo que muestra visualmente el camino que sigue el ordenador para decidir si un producto debe ser rebajado o no.


Requisitos e Instalación

Para ejecutar este proyecto, es necesario disponer de **Python 3.9 o superior**. Se recomienda utilizar un entorno virtual. Las librerías necesarias se instalan con el siguiente comando:

```bash
pip install pandas scikit-learn matplotlib seaborn datasets
```

Guía de Ejecución

Para reproducir los experimentos presentados en la memoria del TFG, los scripts deben ejecutarse estrictamente en este orden desde la terminal:

1. **Obtención y preparación de datos:**
```bash
python Muestras_aleatorias.py
python Variable_Objetivo.py
```

2. **Análisis exploratorio:**
```bash
python Analisis_Exploratorio_de_datos.py
```

3. **Entrenamiento y métricas de los modelos:**
```bash
python Arboles_de_decision.py
python SVM.py
```

4. **Extracción de explicabilidad y gráficas:**
```bash
python Matrices_de_confusion.py
python Importancia_variable.py
python Reglas_Arbol_de_decision.py
python Analisis_Errores.py
python Visualizar_Arbol.py
```
