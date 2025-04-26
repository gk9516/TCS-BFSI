
# 💳 Credit Risk Prediction - German Credit Dataset

## 🛠 Project Description
```markdown
This project implements a machine learning pipeline to predict credit risk based on the German Credit dataset. It includes data preprocessing, feature engineering, model training using Random Forest Classifier, hyperparameter tuning, evaluation metrics, and an interactive Streamlit web application for live predictions.
```

## 🧩 Technologies Used
```markdown
- Python 3.8+
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Seaborn, Matplotlib
- Plotly
- Joblib
```

## 🧪 Key Components
### Data Preprocessing:
```markdown
- Handling missing values
- Feature engineering: Credit per Duration, Age-Job Ratio, Loan Purpose Grouping
- Scaling numeric features
- Encoding categorical variables
```

### Model Building:
```markdown
- Random Forest Classifier
- Hyperparameter Tuning via GridSearchCV
- Feature Importance Analysis
```

### Evaluation Metrics:
```markdown
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix Visualization
```

### Streamlit App:
```markdown
- Tabbed Layout: Data Overview, Model Insights, Prediction Form
- Live prediction with confidence gauge
- Downloadable CSV of results
```

## ⚡ Installation
### Clone the repository:
```bash
git clone https://github.com/your-repo/credit-risk-streamlit.git
cd credit-risk-streamlit
```

### Install the dependencies:
```bash
pip install -r requirements.txt
```

### Start the Streamlit server:
```bash
streamlit run app.py
```

## 📂 File Structure
```bash
├── app.py                   # Main Streamlit App
├── german_credit_data.csv    # Dataset
├── README.md                 # Project Documentation
├── requirements.txt          # Dependencies
└── assets/                   # (Optional) Custom images, logos
```

## 📈 Model Results
```markdown
| Metric      | Score   |
|-------------|---------|
| Accuracy    | 99.5%   |
| Precision   | 99.4%   |
| Recall      | 97.4%   |
| F1-Score    | 98.7%   |

- Top Features: Credit amount, Duration, Age
- Very low misclassification rate
- Robust performance on unseen test data
```

## 🎯 How the App Works
```markdown
- **Data Overview Tab**: View raw dataset, statistics, and missing values
- **Model Insights Tab**: Analyze evaluation metrics and feature importances
- **Prediction Tab**: Input applicant details and get real-time risk prediction with confidence score and downloadable result
```

## 🔮 Future Scope
```markdown
- Add SHAP explainability plots
- Automate periodic model retraining
- Deploy app on AWS/GCP for production use
```

## 👤 Author
```markdown
Made with 💻 and ☕ by Ganesh Kumar
```

[Link to the Application](https://gk9516-tcs-bfsi-credit-risk-app-o2dzp3.streamlit.app/)
