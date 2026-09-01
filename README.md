# Predictive Maintenance Dashboard

An end-to-end machine learning project focused on predictive maintenance using the NASA CMAPSS turbofan engine dataset.

The goal of this project is to predict whether an engine is approaching failure based on operational sensor data and degradation patterns over time.

---

## What this project does

- Analyzes engine sensor data
- Tracks degradation across operational cycles
- Predicts machine failure risk using machine learning
- Visualizes engine health through an interactive Streamlit dashboard
- Explains predictions using SHAP feature importance

![dashboard board](dashboard.png)
---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- SHAP
- Matplotlib

---

## Model Workflow

### Data Processing
- cleaned and prepared sensor data
- created Remaining Useful Life (RUL)
- engineered rolling average features

### Machine Learning
- converted the problem into a classification task
- trained an XGBoost classifier
- evaluated using accuracy, precision, recall and F1-score

### Deployment
Built an interactive Streamlit dashboard with:
- engine selection
- cycle-wise monitoring
- failure probability prediction
- sensor trend visualization
- model explainability

---

## Model Performance

Achieved approximately **96% classification accuracy** on test data.

---

## Dataset

NASA CMAPSS Turbofan Engine Dataset

https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
