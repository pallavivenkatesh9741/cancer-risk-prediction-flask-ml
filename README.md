## Cancer Risk Prediction System using Machine Learning

## 📌 Project Overview
The **Cancer Risk Prediction System** is a machine learning–based web application developed using **Flask** and **Random Forest Classifier**.  
It predicts the **cancer severity level** based on key health and lifestyle factors provided by the user.

This project demonstrates the complete **end-to-end ML workflow**, including data preprocessing, model training, evaluation, and deployment as a web application.

---

## 🎯 Objective
The main goal of this project is to:
- Analyze medical and lifestyle factors
- Predict the cancer risk level accurately
- Provide a simple and interactive web interface for predictions

---

## 🧠 Machine Learning Model
- **Algorithm Used:** Random Forest Classifier
- **Why Random Forest?**
  - Handles non-linear relationships well
  - Reduces overfitting
  - Works efficiently with multiple features
- **Evaluation Metrics:**
  - Accuracy Score
  - Confusion Matrix
  - Classification Report

---

## 📊 Dataset Description
The dataset (`cancer.csv`) contains medical and environmental attributes related to cancer risk.

### Selected Features:
- Air Pollution
- Genetic Risk
- Obesity
- Balanced Diet
- Occupational Hazards
- Coughing of Blood

### Target Variable:
- **Level** (Cancer severity category)

---

## ⚙️ Technology Stack
- **Programming Language:** Python
- **Web Framework:** Flask
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas
- **Frontend:** HTML (Jinja2 Templates)
- **Model:** Random Forest Classifier

---

## 🏗️ Project Structure
-│
-├── app.py
-├── cancer.csv
-├── README.md
-├── templates/
-│ └── index.html
-└── static/
-└── style.css (optional)
