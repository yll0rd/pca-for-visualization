# 📊 Data Cluster Visualizer

An interactive data visualization tool that uses **Principal Component Analysis (PCA)** via **Singular Value Decomposition (SVD)** to transform high-dimensional datasets into intuitive 1D, 2D, and 3D visualizations. The application helps users explore hidden structures, patterns, and clusters within their data through an easy-to-use interface.

---

## 🚀 Features

* 📂 Load datasets from:

  * CSV
  * Excel (.xlsx)
  * JSON
  * SQL Databases

* 🧹 Automated Data Preprocessing

  * Missing value handling
  * Data cleaning
  * Feature selection
  * Data standardization

* 🔬 PCA Using SVD

  * Mathematical implementation of PCA through Singular Value Decomposition
  * User-controlled number of principal components
  * Explained variance analysis

* 📈 Interactive Visualizations

  * 1D PCA projections
  * 2D scatter plots
  * 3D interactive plots
  * Automatic 2D projection for datasets with more than 3 principal components

* 🎛 Interactive User Interface

  * Real-time parameter adjustments
  * Dynamic plot updates
  * Dataset preview and exploration

---

## 🏗 Architecture

```text
Dataset
   │
   ▼
Data Loading
   │
   ▼
Data Cleaning & Preprocessing
   │
   ▼
Feature Standardization
   │
   ▼
SVD Computation
   │
   ▼
PCA Transformation
   │
   ▼
Visualization Layer
   │
   ├── 1D Plot
   ├── 2D Plot
   └── 3D Plot
```

---

## 📚 Mathematical Background

### Step 1: Standardize Data

The dataset is standardized before PCA:

[
Z = \frac{X - \mu}{\sigma}
]

where:

* (X) = Original data
* (\mu) = Mean
* (\sigma) = Standard deviation

---

### Step 2: Singular Value Decomposition

The standardized matrix (Z) is decomposed as:

[
Z = U \Sigma V^T
]

Where:

* (U) = Left singular vectors
* (\Sigma) = Singular values
* (V^T) = Right singular vectors

---

### Step 3: Principal Components

The principal components are obtained from:

[
PC = ZV
]

The directions of maximum variance correspond to the columns of (V).

---

## 🖥 Example Workflow

### Upload Dataset

```text
customer_data.csv
```

### Select Number of Principal Components

```text
2
```

### Generate Visualization

```text
PCA → SVD → 2D Scatter Plot
```

### Explore Clusters

* Customer segments
* Behavioral groups
* Anomaly detection
* Hidden patterns

---

## 📂 Supported File Formats

| Format        | Supported |
| ------------- | --------- |
| CSV           | ✅         |
| Excel (.xlsx) | ✅         |
| JSON          | ✅         |
| SQL Database  | ✅         |

---

## 🎯 Use Cases

### Machine Learning

* Feature reduction
* Data preprocessing
* Model visualization

### Data Science

* Exploratory Data Analysis (EDA)
* Pattern discovery
* Cluster analysis

### Research

* High-dimensional data exploration
* Scientific visualization

### Education

* Learning PCA
* Understanding SVD
* Visualizing dimensionality reduction

---

## 📊 Outputs

The application generates:

* Principal Component Scores
* Explained Variance Ratios
* Cumulative Variance Analysis
* Interactive 1D Visualizations
* Interactive 2D Visualizations
* Interactive 3D Visualizations

---

## ⚙ Installation

```bash
git clone https://github.com/Leonhard-Hopeful/pca-for-visualization.git

cd pca-for-visualization

python3 -m venv venv && source venv/Scripts/activate

pip install -r requirements.txt

streamlit run app.py
```

---

## 📦 Example Requirements

```txt
streamlit
pandas
numpy
scikit-learn
plotly
openpyxl
sqlalchemy
```

---

## 🤝 Contributing

Contributions are welcome. Feel free to open issues, submit feature requests, or create pull requests to improve the project.

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgments

This project was created to provide an intuitive way to understand and visualize high-dimensional data through the mathematical foundations of **PCA** and **Singular Value Decomposition (SVD)**.
