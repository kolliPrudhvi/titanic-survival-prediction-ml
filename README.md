# 🚢 Titanic Survival Prediction – Machine Learning Project

A beginner-friendly machine learning project to predict survival on the Titanic using the [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic). The project includes data cleaning, feature engineering, logistic regression modeling, and evaluation using scikit-learn.

---

## 📂 Project Structure

- **Data Preprocessing**
  - Handle missing values
  - Encode categorical features (`Sex`, `Embarked`)
- **Feature Selection**
  - Selected relevant features like `Pclass`, `Age`, `Fare`, etc.
- **Model**
  - Trained a Logistic Regression model
- **Evaluation**
  - Accuracy score
  - Confusion Matrix
  - Classification Report

---

## 🧠 Key Concepts

- Handling Missing Data (`Age`, `Embarked`)
- Feature Engineering (`Sex` mapping, One-Hot Encoding `Embarked`)
- Train/Test Split
- Logistic Regression with `scikit-learn`
- Model evaluation using accuracy, confusion matrix, and F1-score

---

## 📊 Model Performance

| Metric            | Value   |
|-------------------|---------|
| Training Accuracy | ~0.79   |
| Test Accuracy     | ~0.81   |
| Evaluation        | Precision, Recall, F1-Score (see output) |

---

## 🛠️ How to Run

```bash
# Clone the repo
git clone https://github.com/YOUR-USERNAME/titanic-survival-prediction-ml.git
cd titanic-survival-prediction-ml

# Open and run in Jupyter Notebook or VS Code
