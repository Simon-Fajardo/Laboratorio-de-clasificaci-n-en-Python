# Laboratorio-de-clasification-en-Python
# Bank Marketing Classification with PySpark ML

> **Big Data Processing — Pontificia Universidad Javeriana**  
> Author: Simón Andrés Fajardo Franky  
> Date: April 28 – May 25, 2026

---

## Description

This project implements and compares five Machine Learning classification models using **Apache Spark (PySpark)** on the bank marketing dataset from the UCI repository. The objective is to predict whether a client will subscribe to a term deposit based on direct marketing campaigns (phone calls) conducted by a Portuguese banking institution.

---

## Objective

Apply the complete data processing and classification workflow in a distributed Spark ML environment, including:

- Exploratory Data Analysis (EDA)
- Data cleaning and variable transformation
- Class balancing through oversampling
- Encoding categorical variables
- Training and evaluation of multiple classification models

---

## Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

The dataset contains information related to direct marketing campaigns from a Portuguese bank. The target variable (`y`) indicates whether the client subscribed to a term deposit.

| Variable | Type | Description |
|---|---|---|
| `age` | Integer | Client age |
| `job` | Categorical | Job type |
| `marital` | Categorical | Marital status |
| `education` | Categorical | Education level |
| `default` | Binary | Has credit in default? |
| `balance` | Integer | Average yearly balance (euros) |
| `housing` | Binary | Has housing loan? |
| `loan` | Binary | Has personal loan? |
| `contact` | Categorical | Contact type (cellular/telephone) |
| `duration` | Integer | Duration of the last call (seconds) |
| `campaign` | Integer | Number of contacts during this campaign |
| `pdays` | Integer | Days since the previous contact |
| `previous` | Integer | Number of contacts before this campaign |
| `poutcome` | Categorical | Outcome of the previous campaign |
| `y` | Binary | **Target variable** — Did the client subscribe to the deposit? |

---

## Notebook Structure

```text
Lab_Clasification_Fajardo.ipynb
│
├── 1. Spark Session Configuration
├── 2. Data Loading from HDFS
├── 3. Dataset Understanding and Description
│   ├── Schema and data types
│   ├── Descriptive statistics
│   └── Target variable distribution
├── 4. Exploratory Data Analysis (EDA)
│   ├── Histograms of numerical variables
│   ├── Boxplots by target class
│   ├── Correlation matrix (Pearson)
│   ├── Pairplot
│   └── Categorical variable analysis
├── 5. Data Cleaning and Processing
│   ├── Null value verification
│   ├── Outlier filtering (PREVIOUS > 30)
│   └── Removal of PDAYS column
├── 6. Class Balancing (Oversampling)
├── 7. Feature Engineering
│   ├── StringIndexer + OneHotEncoder for categorical variables
│   ├── VectorAssembler for feature vector construction
│   └── Transformation pipeline
├── 8. Train/Test Split (80/20)
└── 9. Classification Models
    ├── Logistic Regression
    ├── Decision Tree
    ├── Random Forest
    ├── Gradient Boosted Tree (GBT)
    └── Support Vector Machine (SVM)
```

---

## Implemented Models

| Model | PySpark Library |
|---|---|
| Logistic Regression | `pyspark.ml.classification.LogisticRegression` |
| Decision Tree | `pyspark.ml.classification.DecisionTreeClassifier` |
| Random Forest | `pyspark.ml.classification.RandomForestClassifier` |
| Gradient Boosted Tree | `pyspark.ml.classification.GBTClassifier` |
| Support Vector Machine | `pyspark.ml.classification.LinearSVC` |

Each model is evaluated using:

- **Confusion Matrix**
- **ROC Curve and AUC**
- **Accuracy, Precision, Recall, and F1-Score**

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC ROC |
|---|---|---|---|---|---|
| Gradient Boosted Tree | **~85.8%** | **~85.9%** | **~85.8%** | **~85.8%** | **~0.93** |
| Support Vector Machine | ~85%+ | ~85%+ | ~85%+ | ~85%+ | ~0.93 |
| Logistic Regression | ~80%+ | ~80%+ | ~80%+ | ~80%+ | ~0.91 |
| Random Forest | ~80%+ | ~80%+ | ~80%+ | ~80%+ | ~0.76 |
| Decision Tree | ~80%+ | ~80%+ | ~80%+ | ~80%+ | ~0.76 |

> **Best Model: Gradient Boosted Tree (GBT)** — Highest AUC and most balanced performance metrics.

---

## Main Findings

- The variable `duration` (call duration) showed the highest correlation with the target variable (0.39).
- The dataset presented a strong class imbalance: 88.3% `"no"` vs. 11.7% `"yes"`, corrected through oversampling.
- Removing the `pdays` column (81%+ of records with value -1) and filtering `previous > 30` improved dataset quality.
- Variables such as `day`, `age`, and `balance` showed weak linear correlation with the target variable.

---

## Technologies and Libraries

- **Apache Spark / PySpark** — Distributed processing and Machine Learning
- **Python 3.x**
- **pandas** — DataFrame manipulation
- **NumPy** — Matrix algebra
- **Matplotlib / Seaborn** — Visualization
- **scikit-learn** — ROC curve calculations (`roc_curve`, `auc`)
- **findspark** — Spark environment initialization

---

## Environment Requirements

This notebook is designed to run on a **Spark cluster** with HDFS access. It requires:

- Apache Spark configured with a Master node on the local network
- HDFS with the file `bank-full.csv` located in the `/csv/` directory
- FAIR Scheduler configured with `fairscheduler.xml`
- Python with all required libraries installed

> **Note:** The Spark Master IP (`spark://10.43.97.187:7077`) and the HDFS address (`hdfs://10.195.34.34:9000`) are specific to the university environment and should be adapted according to the available cluster.

---

## References

- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems.

---

## Author

**Simón Andrés Fajardo Franky**  
Pontificia Universidad Javeriana — Big Data Processing  
2026
