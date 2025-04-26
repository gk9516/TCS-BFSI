# 💳 Credit Risk Prediction - German Credit Dataset

## 🛠 Project Description
This project implements a machine learning pipeline to predict credit risk based on the German Credit dataset. It includes data preprocessing, feature engineering, model training using Random Forest Classifier, hyperparameter tuning, evaluation metrics, and an interactive **Streamlit** web application for live predictions.

---

## 🧩 Technologies Used
- Python 3.8+
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Seaborn, Matplotlib
- Plotly
- Joblib

---

## 🧪 Key Components
- **Data Preprocessing:** 
  - Handling missing values
  - Feature engineering: Credit per Duration, Age-Job Ratio, Loan Purpose Grouping
  - Scaling numeric features
  - Encoding categorical variables

- **Model Building:**
  - Random Forest Classifier
  - Hyperparameter Tuning via GridSearchCV
  - Feature Importance Analysis
  
- **Evaluation Metrics:**
  - Classification Report (Precision, Recall, F1-score)
  - Confusion Matrix Visualization

- **Streamlit App:**
  - Tabbed Layout: Data Overview, Model Insights, Prediction Form
  - Live prediction with confidence gauge
  - Downloadable CSV of results

---

## ⚡ Installation
Clone the repository:
```bash
git clone https://github.com/your-repo/credit-risk-streamlit.git
cd credit-risk-streamlit
