# 📌 Customer Churn Prediction using ANN

## 📖 Project Overview

Customer churn is a major challenge for subscription-based businesses, where retaining existing customers is often more cost-effective than acquiring new ones.

In this project, I built an **Artificial Neural Network (ANN)** model to predict whether a customer is likely to churn based on their demographic, contract, and service-related information. The goal is to **identify high-risk customers** and provide insights to improve retention strategies.

---

## 🎯 Objectives

* Understand factors influencing customer churn.
* Build and train an ANN model to predict churn probability.
* Compare model performance with baseline machine learning models.
* Provide actionable insights for customer retention.

---

## 📊 Dataset

* **Source**: [Telco Customer Churn Dataset – Kaggle]
* **Lines**: 3192
* **Target Variable**: `Churn` (Yes/No)
* **Features**: Customer demographics, contract type, tenure, internet service, payment method, etc.

---

## 🛠️ Tools & Technologies

* **Python**: Pandas, NumPy, Matplotlib, Seaborn
* **Deep Learning**: TensorFlow / Keras (ANN Model)
* **Machine Learning**: Scikit-learn (for comparison models like Logistic Regression)
* **Visualization**: Matplotlib, Seaborn
* **Database**: SQL (for exploratory queries)

---

## 🔍 Methodology

1. **Data Preprocessing**

   * Handled missing values.
   * Encoded categorical variables (One-Hot Encoding).
   * Scaled numerical features for ANN training.

2. **Exploratory Data Analysis (EDA)**

   * Churn rate by contract type, internet service, and payment method.
   * Correlation analysis of features with churn.

3. **Model Building**

   * ANN with input layer, 2 hidden layers (ReLU activation), and output layer (Sigmoid).
   * Optimizer: Adam | Loss: Binary Crossentropy | Metrics: Accuracy.

4. **Evaluation**

   * Train/Test split: 80/20.
   * Achieved \~85% accuracy and good ROC-AUC score.
   * Compared with Logistic Regression and Random Forest baselines.

---

## 📈 Key Insights

* **Month-to-month contracts** have the highest churn risk (\~45%).
* Customers with **electronic check payment method** are more likely to churn.
* Longer tenure significantly reduces churn probability.
* ANN performed better than traditional models, capturing complex non-linear patterns.

---

## 📂 Repository Structure

```
Customer-Churn-ANN/
│── data/                # Dataset (if permissible, else link to source)
│── notebooks/           # Jupyter notebooks for EDA & modeling
│── models/              # Saved ANN model files
│── images/              # Graphs and plots
│── Customer_Churn_ANN.ipynb
│── requirements.txt     # Dependencies
│── README.md
```

---

