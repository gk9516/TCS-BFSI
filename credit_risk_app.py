import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import plotly.graph_objects as go
import base64

# ---- Page Config ----
st.set_page_config(page_title="Credit Risk Classifier", layout="wide", page_icon="💳")

# ---- Dark Neon Theme ----
neon_style = """
<style>
body {
    background-color: #0e1117;
    color: #39FF14;
}
h1, h2, h3, h4, h5, h6 {
    color: #39FF14;
}
.stButton>button {
    color: black;
    background-color: #39FF14;
    border-radius: 8px;
}
div.stSlider > div[data-baseweb="slider"] > div {
    background: #39FF14;
}
div[data-testid="stSelectbox"] select {
    background-color: #1c1f26;
    color: #39FF14;
}
input[type="number"] {
    background-color: #1c1f26;
    color: #39FF14;
}
</style>
"""
st.markdown(neon_style, unsafe_allow_html=True)

# ---- Group Purpose ----
def group_purpose(p):
    p = p.lower()
    if "car" in p:
        return "vehicle"
    elif "business" in p:
        return "business"
    elif "education" in p:
        return "education"
    elif "radio" in p or "tv" in p or "furniture" in p or "equipment" in p or "domestic appliances" in p:
        return "appliance"
    elif "repairs" in p:
        return "repair"
    elif "vacation" in p or "others" in p:
        return "leisure"
    else:
        return "other"

@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv", delimiter=",")
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

def preprocess(df):
    df = df.copy()
    df['Credit_Risk'] = np.where(df['Purpose'].str.lower().str.contains("radio|tv|furniture|car"), 1, 0)

    df['Credit_Per_Duration'] = df['Credit amount'] / (df['Duration'] + 1)
    df['Age_Job_Ratio'] = df['Age'] / (df['Job'] + 1)
    df['Is_Young'] = (df['Age'] < 25).astype(int)
    df['Is_Large_Loan'] = (df['Credit amount'] > df['Credit amount'].quantile(0.75)).astype(int)

    df["Purpose_Grouped"] = df["Purpose"].apply(group_purpose)
    df.drop(columns=["Purpose"], inplace=True)

    y = df['Credit_Risk']
    X = df.drop(columns=["Credit_Risk"])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])
    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns, scaler

# ---- Load & Preprocess ----
df = load_data()
X, y, feature_names, scaler = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
grid = GridSearchCV(model, {'n_estimators': [100], 'max_depth': [10, None]}, cv=5, scoring='f1')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# ---- Tabs Layout ----
tab1, tab2, tab3 = st.tabs(["📁 Data Overview", "📊 Model Insights", "🧪 Try it Yourself"])

# ---- Tab 1: Data ----
with tab1:
    st.header("📁 Raw Dataset")
    st.dataframe(df.head())

    st.subheader("📌 Summary Stats")
    st.write(df.describe())

    st.subheader("📌 Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("📌 Average Credit by Job")
    st.bar_chart(df.groupby("Job")["Credit amount"].mean())

# ---- Tab 2: Insights ----
with tab2:
    st.header("📊 Model Evaluation")
    st.write("### Classification Report")
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Greens", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.write("### Feature Importances")
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=False).head(10)

    fig2, ax2 = plt.subplots()
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
    ax2.set_title("Top 10 Features")
    st.pyplot(fig2)

# ---- Tab 3: Prediction ----
with tab3:
    st.header("🧪 Predict Credit Risk")

    with st.form("user_form"):
        st.write("Enter applicant info:")
        inputs = {}
        for col in df.columns:
            if col in ["Credit_Risk", "Purpose"]:
                continue
            if df[col].dtype == "object":
                inputs[col] = st.selectbox(col, df[col].dropna().unique())
            else:
                inputs[col] = st.number_input(
                    col, 
                    min_value=int(df[col].min()), 
                    max_value=int(df[col].max()), 
                    value=int(df[col].mean()),
                    step=1
                )

        selected_purpose = st.selectbox("Purpose", df["Purpose"].dropna().unique())
        inputs["Credit_Per_Duration"] = inputs["Credit amount"] / (inputs["Duration"] + 1)
        inputs["Age_Job_Ratio"] = inputs["Age"] / (inputs["Job"] + 1)
        inputs["Is_Young"] = int(inputs["Age"] < 25)
        inputs["Is_Large_Loan"] = int(inputs["Credit amount"] > df["Credit amount"].quantile(0.75))
        inputs["Purpose_Grouped"] = group_purpose(selected_purpose)

        submitted = st.form_submit_button("Predict 🔍")

    if submitted:
        input_df = pd.DataFrame([inputs])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        input_scaled = scaler.transform(input_df)

        pred = best_model.predict(input_scaled)[0]
        prob = best_model.predict_proba(input_scaled)[0][1]

        st.markdown("## 🎯 Prediction Result")
        st.success("✅ Good Credit Risk" if pred == 1 else "❌ Bad Credit Risk")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob*100),
            title={'text': "Confidence (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#39FF14"},
                   'steps': [
                       {'range': [0, 50], 'color': "red"},
                       {'range': [50, 75], 'color': "orange"},
                       {'range': [75, 100], 'color': "green"}]
                   }
        ))
        st.plotly_chart(fig)

        result_df = pd.DataFrame([inputs])
        result_df["Prediction"] = ["Good" if pred else "Bad"]
        result_df["Confidence"] = round(prob*100, 2)

        csv = result_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f"""
            <a href="data:file/csv;base64,{b64}" download="credit_prediction.csv">
                📥 Download Prediction as CSV
            </a>
        """, unsafe_allow_html=True)
