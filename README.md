# Laboratorio-de-clasificacion-en-Python
# 🏦 Bank Marketing Classification with PySpark ML

> **Big Data Processing — Pontificia Universidad Javeriana**  
> Autor: Simón Andrés Fajardo Franky  
> Fecha: Abril 28 – Mayo 25, 2026

---

## 📋 Descripción

Este proyecto implementa y compara cinco modelos de clasificación de Machine Learning usando **Apache Spark (PySpark)** sobre el dataset de marketing bancario del repositorio UCI. El objetivo es predecir si un cliente suscribirá un depósito a término fijo, a partir de campañas de marketing directo (llamadas telefónicas) realizadas por una institución bancaria portuguesa.

---

## 🎯 Objetivo

Aplicar el proceso completo de tratamiento de datos y modelos de clasificación en un entorno distribuido con Spark ML, incluyendo:

- Exploración y análisis de datos (EDA)
- Limpieza y transformación de variables
- Balanceo de clases mediante oversampling
- Codificación de variables categóricas
- Entrenamiento y evaluación de múltiples modelos de clasificación

---

## 📊 Dataset

**Fuente:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

El dataset contiene información sobre campañas de marketing directo de un banco portugués. La variable objetivo (`y`) indica si el cliente suscribió o no un depósito a término fijo.

| Variable | Tipo | Descripción |
|---|---|---|
| `age` | Integer | Edad del cliente |
| `job` | Categorical | Tipo de trabajo |
| `marital` | Categorical | Estado civil |
| `education` | Categorical | Nivel educativo |
| `default` | Binary | ¿Tiene crédito en mora? |
| `balance` | Integer | Saldo promedio anual (euros) |
| `housing` | Binary | ¿Tiene préstamo de vivienda? |
| `loan` | Binary | ¿Tiene préstamo personal? |
| `contact` | Categorical | Tipo de contacto (cellular/telephone) |
| `duration` | Integer | Duración de la última llamada (segundos) |
| `campaign` | Integer | Número de contactos en esta campaña |
| `pdays` | Integer | Días desde el último contacto previo |
| `previous` | Integer | Contactos anteriores a esta campaña |
| `poutcome` | Categorical | Resultado de la campaña anterior |
| `y` | Binary | **Variable objetivo** — ¿Suscribió el depósito? |

---

## 🗂️ Estructura del Notebook

```
Lab_Clasification_Fajardo.ipynb
│
├── 1. Configuración de la sesión Spark
├── 2. Carga de datos desde HDFS
├── 3. Comprensión y descripción del dataset
│   ├── Schema y tipos de datos
│   ├── Estadísticas descriptivas
│   └── Distribución de la variable objetivo
├── 4. Análisis Exploratorio de Datos (EDA)
│   ├── Histogramas de variables numéricas
│   ├── Boxplots por clase objetivo
│   ├── Matriz de correlación (Pearson)
│   ├── Pairplot
│   └── Análisis de variables categóricas
├── 5. Limpieza y tratamiento de datos
│   ├── Verificación de valores nulos
│   ├── Revisión y filtrado de outliers (PREVIOUS > 30)
│   └── Eliminación de columna PDAYS
├── 6. Balanceo de clases (Oversampling)
├── 7. Feature Engineering
│   ├── StringIndexer + OneHotEncoder para variables categóricas
│   ├── VectorAssembler para construcción del vector de features
│   └── Pipeline de transformación
├── 8. División Train/Test (80/20)
└── 9. Modelos de clasificación
    ├── Logistic Regression
    ├── Decision Tree
    ├── Random Forest
    ├── Gradient Boosted Tree (GBT)
    └── Support Vector Machine (SVM)
```

---

## 🤖 Modelos Implementados

| Modelo | Librería PySpark |
|---|---|
| Logistic Regression | `pyspark.ml.classification.LogisticRegression` |
| Decision Tree | `pyspark.ml.classification.DecisionTreeClassifier` |
| Random Forest | `pyspark.ml.classification.RandomForestClassifier` |
| Gradient Boosted Tree | `pyspark.ml.classification.GBTClassifier` |
| Support Vector Machine | `pyspark.ml.classification.LinearSVC` |

Cada modelo es evaluado con:
- **Matriz de Confusión**
- **Curva ROC y AUC**
- **Accuracy, Precision, Recall y F1-Score**

---

## 📈 Resultados

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC ROC |
|---|---|---|---|---|---|
| Gradient Boosted Tree | **~85.8%** | **~85.9%** | **~85.8%** | **~85.8%** | **~0.93** |
| Support Vector Machine | ~85%+ | ~85%+ | ~85%+ | ~85%+ | ~0.93 |
| Logistic Regression | ~80%+ | ~80%+ | ~80%+ | ~80%+ | ~0.91 |
| Random Forest | ~80%+ | ~80%+ | ~80%+ | ~80%+ | ~0.76 |
| Decision Tree | ~80%+ | ~80%+ | ~80%+ | ~80%+ | ~0.76 |

> ✅ **Mejor modelo: Gradient Boosted Tree (GBT)** — Mayor AUC y métricas más balanceadas.

---

## 🔑 Hallazgos Principales

- La variable `duration` (duración de la llamada) es la que mayor correlación presenta con la variable objetivo (0.39).
- El dataset presenta un **fuerte desbalance de clases**: 88.3% "no" vs. 11.7% "yes", corregido con oversampling.
- La eliminación de la columna `pdays` (81%+ de registros con valor -1) y el filtrado de `previous > 30` mejoraron la calidad del dataset.
- Variables como `day`, `age` y `balance` presentan correlación lineal débil con la variable objetivo.

---

## 🛠️ Tecnologías y Librerías

- **Apache Spark / PySpark** — Procesamiento distribuido y ML
- **Python 3.x**
- **pandas** — Manipulación de DataFrames
- **NumPy** — Álgebra matricial
- **Matplotlib / Seaborn** — Visualización
- **scikit-learn** — Cálculo de curvas ROC (`roc_curve`, `auc`)
- **findspark** — Inicialización del entorno Spark

---

## ⚙️ Requisitos de Entorno

Este notebook está diseñado para ejecutarse en un **clúster Spark** con acceso a HDFS. Requiere:

- Apache Spark configurado con un Master en la red local
- HDFS con el archivo `bank-full.csv` en la ruta `/csv/`
- Scheduler FAIR configurado con `fairscheduler.xml`
- Python con las librerías listadas arriba instaladas

> ⚠️ **Nota:** La IP del Master Spark (`spark://10.43.97.187:7077`) y la dirección HDFS (`hdfs://10.195.34.34:9000`) son específicas del entorno universitario. Deben ajustarse según el clúster disponible.

---

## 📚 Referencias

- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [PySpark ML Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems.

---

## 👤 Autor

**Simón Andrés Fajardo Franky**  
Pontificia Universidad Javeriana — Big Data Processment  
2026
