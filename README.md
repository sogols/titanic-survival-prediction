# 🚢 Titanic Survival Prediction – Machine Learning Project

This project applies various machine learning models to predict passenger survival on the Titanic, using the famous [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic). It includes data preprocessing, visualization, multiple ML models, deep learning with MLP, hyperparameter tuning, and final model comparison.

---

## 📁 Project Structure

- `titanic_analysis_visualization.ipynb` – EDA, feature engineering, preprocessing
- `titanic_survival_prediction.ipynb` – ML models, deep learning, tuning, evaluation
- `titanic_cleaned.csv` – Preprocessed dataset used for modeling
- `requirements.txt` – List of required Python packages
- `README.md` – This file

---

## 🔍 Project Goals

- Understand the Titanic dataset and clean missing or noisy data
- Build and compare classification models:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - K-Nearest Neighbors
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Multi-Layer Perceptron (MLP – neural network)
- Tune hyperparameters using `Keras Tuner`
- Evaluate models with accuracy, precision, recall, F1 score, and confusion matrix

---

## 🧠 Best Performing Models

| Model               | Accuracy | F1 Score |
|---------------------|----------|----------|
| Random Forest       | 0.8268   | 0.79     |
| KNN                 | 0.8212   | 0.78     |
| MLP (Tuned)         | 0.8156   | 0.76     |
| MLP (Deep + Tuned)  | 0.8045   | 0.72     |

---

## 📌 Dependencies

Install them with:

```bash
pip install -r requirements.txt
