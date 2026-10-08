# 🌱 Dry Bean Variety Classification

> End-to-end machine learning classification project for identifying dry bean varieties from morphological measurements using multiple supervised learning algorithms, with the best-performing model deployed through Streamlit.

**Academic Project:** M.Tech Data Science & Engineering  
**Dataset:** UCI Dry Bean Dataset  
**Deployment:** Streamlit

---

## 📌 Overview

The objective of this project is to classify dry bean varieties based on **16 morphological characteristics** extracted from bean images.

The project follows an end-to-end machine learning workflow covering data quality assessment, exploratory data analysis, model development, evaluation, model selection, serialization, and deployment.

Five supervised classification algorithms were trained and evaluated, with **Random Forest** selected as the final deployment model.

### 🔄 Machine Learning Workflow

```text
Data Loading
     ↓
Data Quality Assessment
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Preparation
     ↓
Model Training
     ↓
Model Evaluation & Comparison
     ↓
Random Forest Selection
     ↓
Model Serialization
     ↓
Streamlit Deployment
     ↓
Deployment Validation
📊 Key Results

Five classification models were trained and evaluated on the held-out test dataset.

Model	Test Accuracy
Logistic Regression	91.92%
Decision Tree	89.66%
K-Nearest Neighbors	91.55%
Gaussian Naive Bayes	76.30%
Random Forest	92.03%

🏆 Final Model Performance

The Random Forest Classifier was selected as the deployment model and achieved:

Metric	Result
Test Accuracy	92.03%
ROC-AUC	0.9923
F1-Score	0.9202
Matthews Correlation Coefficient	0.9035

The trained model was serialized using joblib and integrated into the Streamlit application.


📁 Dataset

The project uses the UCI Dry Bean Dataset, which contains morphological measurements for seven different dry bean varieties.

Dataset Summary
Property	Value
Original Records	13,611
Duplicate Records Identified	68
Cleaned Records	13,543
Input Features	16
Target Classes	7
🎯 Target Classes
SEKER
BARBUNYA
BOMBAY
CALI
HOROZ
SIRA
DERMASON


🔢 Input Features

The model uses the following 16 morphological characteristics:

Area
Perimeter
MajorAxisLength
MinorAxisLength
AspectRation
Eccentricity
ConvexArea
EquivDiameter
Extent
Solidity
roundness
Compactness
ShapeFactor1
ShapeFactor2
ShapeFactor3
ShapeFactor4


🔬 Machine Learning Workflow

1. Data Quality Assessment

The dataset was examined for:

Missing values
Duplicate records
Data types
Feature distributions
Class distribution
Potentially redundant features

No missing values were identified. 68 duplicate records were detected and removed during data cleaning.

2. Exploratory Data Analysis

Exploratory analysis included:

Univariate feature analysis
Feature distribution analysis
Class distribution analysis
Correlation analysis
Feature relationship analysis
Identification of highly correlated variables

Examples of strong correlations observed in the dataset include:

Area ↔ ConvexArea
Compactness ↔ ShapeFactor3
AspectRation ↔ Compactness

These relationships were examined during the exploratory analysis to better understand the feature space.

3. Model Training

Five supervised machine learning algorithms were implemented and evaluated:

Logistic Regression
Decision Tree
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Random Forest
4. Model Evaluation

The models were evaluated using multiple classification metrics:

Accuracy
Precision
Recall
F1-score
Matthews Correlation Coefficient
ROC-AUC where applicable
Confusion Matrix

Using multiple metrics provided a more comprehensive assessment of model performance rather than relying only on accuracy.

5. Model Selection

The Random Forest Classifier was selected as the final deployment model based on its overall classification performance across the evaluated metrics.


💾 Model Serialization & Validation

The trained models were serialized using the joblib library.

The final deployment model is stored at:

models/random_forest_model.pkl

The saved Random Forest model was subsequently reloaded and evaluated to verify that serialization did not change its predictions.

Original accuracy: 0.920265780730897
Loaded model accuracy: 0.920265780730897
Predictions identical: True

This confirms that the serialized model can be reliably loaded and used for inference.


🚀 Streamlit Application

The trained Random Forest model was deployed through an interactive Streamlit application.

The application provides two main workflows.


🔹 Single Bean Prediction

Users can enter the 16 morphological measurements of a dry bean and obtain a predicted bean variety.

User Input
    ↓
16 Morphological Features
    ↓
Random Forest Model
    ↓
Predicted Bean Variety

🔹 Batch Prediction

Users can upload a CSV file containing the required input features.

The application:

Validates the required columns.
Generates predictions using the trained Random Forest model.
Compares predictions with actual labels when available.
Calculates prediction accuracy.
Displays prediction results.
Allows prediction results to be downloaded as a CSV file.

✅ Deployment Validation

The Streamlit application was validated using the held-out test dataset used during model evaluation.

Validation Results
Prediction Accuracy: 92.03%
Correct Predictions: 2,493

The deployed application's accuracy matched the notebook evaluation:

Notebook Test Accuracy : 92.03%
Streamlit Accuracy     : 92.03%

This consistency confirms successful integration of the serialized Random Forest model into the deployed application.

🗂️ Repository Structure
dry-bean-variety-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── test_data.csv
│
├── notebooks/
│   └── dry_bean_classification.ipynb
│
└── models/
    ├── decision_tree_model.pkl
    ├── knn_pipeline.pkl
    ├── logistic_pipeline.pkl
    ├── naive_bayes_model.pkl
    └── random_forest_model.pkl
📄 File Description
File / Folder	Purpose
app.py	Streamlit application for model inference
notebooks/	Jupyter Notebook containing data analysis, preprocessing, training, and evaluation
models/	Serialized trained machine learning models
test_data.csv	Held-out test data used for validation
requirements.txt	Python dependencies required to run the application
.gitignore	Files excluded from version control
README.md	Project documentation


🛠️ Tech Stack
Programming & Data
Python
Pandas
NumPy
Machine Learning
Scikit-learn
Logistic Regression
Decision Tree
K-Nearest Neighbors
Gaussian Naive Bayes
Random Forest
Deployment
Streamlit
Joblib
Development & Version Control
Jupyter Notebook
VS Code
Git
GitHub

⚙️ Installation
1. Clone the Repository
git clone https://github.com/armansiddiquee/dry-bean-variety-classification.git
2. Navigate to the Project Directory
cd dry-bean-variety-classification
3. Install Dependencies
pip install -r requirements.txt
▶️ Run the Streamlit Application

Start the application using:

streamlit run app.py

The application will open in your default web browser.

You can then:

Enter individual dry bean measurements.
Generate a predicted bean variety.
Upload a CSV dataset.
Evaluate batch predictions.
Download prediction results.


🔁 Reproducibility

The project maintains a clear separation between model experimentation and deployment.

🧪 Experimentation

The Jupyter Notebook contains:

Dataset loading
Data quality assessment
Data cleaning
Exploratory data analysis
Feature analysis
Model training
Model evaluation
Model comparison

🚀 Deployment

The Streamlit application loads the serialized Random Forest model:

models/random_forest_model.pkl

The model is loaded at application startup rather than being retrained every time the application runs.

This separation provides a reproducible inference workflow and keeps model training independent from application execution.


🔮 Future Improvements

Potential improvements to further productionize the project include:

Hyperparameter optimization for the final Random Forest model
Cross-validation-based model comparison
Feature selection and dimensionality reduction
Experiment tracking with MLflow
Docker-based containerization
Automated testing and CI/CD
Cloud deployment
Model monitoring
Data and model drift detection
API-based model serving

🎓 Academic Context

This project was developed as part of the M.Tech Data Science & Engineering coursework.

The repository presents the work in a production-oriented format to demonstrate the complete machine learning lifecycle, from data preparation and model evaluation through model serialization and application deployment.




👤 Author

Arman Siddiquee

Data Science & Machine Learning

GitHub: @armansiddiquee
