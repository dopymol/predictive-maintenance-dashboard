Predictive Maintenance Dashboard using Machine Learning
Overview

This project is an AI-powered Predictive Maintenance System built using Machine Learning and Streamlit.
The application predicts the failure risk of industrial aircraft engines using real-time sensor telemetry data from the NASA Turbofan Engine Degradation Simulation Dataset.

The system analyzes engine degradation patterns and predicts whether an engine is operating in a:

Healthy State
Warning State
Critical Failure State

The project demonstrates an end-to-end machine learning workflow including:

data preprocessing
feature engineering
model training
evaluation
explainable AI
deployment-ready dashboard development
Problem Statement

Unexpected industrial equipment failure can result in:

production downtime
increased maintenance costs
operational inefficiency
safety risks

This project aims to predict machine failure before breakdown using sensor-based machine learning models.

Dataset

Dataset Used:
NASA C-MAPSS Turbofan Engine Dataset

The dataset contains:

engine operational cycles
multiple sensor readings
degradation progression over time

Source:
NASA Prognostics Data Repository

Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
XGBoost
SHAP
Streamlit
Machine Learning Workflow
1. Data Preprocessing
cleaned raw sensor data
removed non-informative features
handled engine lifecycle sequences
2. Feature Engineering

Created:

Remaining Useful Life (RUL)
rolling average sensor features
degradation indicators
3. Classification Modeling

Converted predictive maintenance into a classification problem:

0 → Healthy
1 → Failure Risk
4. Model Training

Trained:

XGBoost Classifier
5. Model Evaluation

Evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

Achieved approximately:

96% classification accuracy
Streamlit Dashboard Features

The dashboard provides:

Engine selection
Cycle-wise analysis
Failure probability prediction
Sensor trend visualization
Engine lifecycle monitoring
SHAP-based model explainability
Health status categorization
Explainable AI (SHAP)

Integrated SHAP explainability to identify:

which sensors contribute most to failure prediction
feature importance for model decisions

This improves model interpretability and industrial trustworthiness.

Project Structure
Predictive-Maintenance-Dashboard/
│
├── app/
│   └── app.py
│
├── data/
│   └── train_FD001.txt
│
├── models/
│   ├── predictive_maintenance_model.pkl
│   └── feature_columns.pkl
│
├── requirements.txt
├── README.md
├── .gitignore
How to Run Locally
Clone Repository
git clone https://github.com/YOUR_USERNAME/predictive-maintenance-dashboard.git
Install Dependencies
pip install -r requirements.txt
Run Streamlit App
streamlit run app/app.py
Future Improvements

Potential future enhancements:

real-time sensor streaming
cloud deployment
advanced deep learning models (LSTM)
automated maintenance scheduling
MLOps pipeline integration
Business Impact

This solution can help industries:

reduce maintenance costs
prevent unexpected machine failures
improve operational efficiency
optimize maintenance scheduling

Author

Developed as an end-to-end Machine Learning capstone project focused on industrial AI and predictive analytics.