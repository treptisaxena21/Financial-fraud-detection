import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(page_title="Financial Fraud Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ AI-Powered Financial Fraud Detection")
st.write(
    "Train and compare Logistic Regression, Random Forest, and an Artificial Neural Network (ANN) "
    "using the public credit-card fraud dataset."
)
st.warning(
    "Educational demonstration only. Do not use this app to make real financial decisions. "
    "Predictions are not proof that a transaction is fraudulent."
)

with st.sidebar:
    st.header("Settings")
   uploaded_file = st.file_uploader("Upload creditcard.csv (optional)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = pd.read_csv("creditcard.csv")
    
    sample_limit = st.selectbox(
        "Maximum rows to use (faster training)",
        [20000, 50000, 100000, 284807],
        index=2
    )
    test_size = st.slider("Test-set percentage", min_value=10, max_value=40, value=20, step=5)
    train_button = st.button("Train models", type="primary", use_container_width=True)

st.markdown("### 1. Upload your dataset")
st.write("Expected file: `creditcard.csv`, with a target column named `Class` (0 = legitimate, 1 = fraud).")
st.markdown("### 2. Train and compare the models")

if uploaded_file is None:
    st.info("Upload `creditcard.csv` in the sidebar to begin. Get it from the Kaggle Credit Card Fraud Detection dataset.")
    st.link_button(
        "Open dataset on Kaggle",
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    )
    st.stop()


try:
    raw_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

# Clean column names
raw_df.columns = [
    str(col).replace("\ufeff", "").strip()
    for col in raw_df.columns
]

# Find the target column regardless of capitalization
class_columns = [
    col for col in raw_df.columns
    if col.casefold() == "class"
]

if not class_columns:
    st.error(
        "Target column 'Class' was not found. "
        f"Columns in your file: {list(raw_df.columns)}"
    )
    st.stop()

# Rename target column to the exact name used throughout the app
raw_df = raw_df.rename(columns={class_columns[0]: "Class"})

# Remove duplicate column names
raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()].copy()


if raw_df.empty or raw_df["Class"].nunique() < 2:
    st.error("The dataset must contain examples of both classes: 0 (legitimate) and 1 (fraud).")
    st.stop()

raw_df = raw_df.dropna(subset=["Class"]).copy()
raw_df["Class"] = pd.to_numeric(raw_df["Class"], errors="coerce")
raw_df = raw_df.dropna(subset=["Class"])
raw_df = raw_df[raw_df["Class"].isin([0, 1])]
raw_df["Class"] = raw_df["Class"].astype(int)


if len(raw_df) > sample_limit:
    # Stratified sampling keeps the Class column intact
    raw_df, _ = train_test_split(
        raw_df,
        train_size=sample_limit,
        random_state=42,
        stratify=raw_df["Class"]
    )

raw_df = raw_df.reset_index(drop=True)

# Verify that the target column still exists
if "Class" not in raw_df.columns:
    st.error("Class column is missing after sampling. Please check your dataset.")
    st.stop()


st.caption(f"Rows loaded for this run: {len(raw_df):,}")
m1, m2, m3 = st.columns(3)
m1.metric("Transactions", f"{len(raw_df):,}")
m2.metric("Fraud cases", f"{int((raw_df['Class'] == 1).sum()):,}")
m3.metric("Fraud percentage", f"{100 * raw_df['Class'].mean():.3f}%")

with st.expander("Preview dataset"):
    st.dataframe(raw_df.head(10), use_container_width=True)

if train_button:
    with st.spinner("Training models. This can take a few minutes on larger datasets..."):
        X = raw_df.drop(columns=["Class"]).select_dtypes(include=[np.number])
        y = raw_df["Class"]

        if X.empty:
            st.error("No numeric feature columns were found.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42, stratify=y
        )

        models = {
            "Logistic Regression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=42
                ))
            ]),
            "Random Forest": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            "ANN (MLP Neural Network)": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=30,
                    early_stopping=True,
                    random_state=42
                ))
            ])
        }

        results = []
        fitted_models = {}
        predictions = {}
        probabilities = {}

        progress = st.progress(0)
        for i, (name, model) in enumerate(models.items(), start=1):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            try:
                proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
            except Exception:
                proba = pred.astype(float)
                auc = np.nan

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1-score": f1_score(y_test, pred, zero_division=0),
                "ROC-AUC": auc
            })
            fitted_models[name] = model
            predictions[name] = pred
            probabilities[name] = proba
            progress.progress(i / len(models))

        st.session_state["fraud_results"] = pd.DataFrame(results)
        st.session_state["fraud_models"] = fitted_models
        st.session_state["fraud_X_test"] = X_test
        st.session_state["fraud_y_test"] = y_test
        st.session_state["fraud_predictions"] = predictions
        st.session_state["fraud_probabilities"] = probabilities
        st.session_state["fraud_feature_names"] = list(X.columns)
        st.session_state["fraud_sample"] = raw_df

if "fraud_results" not in st.session_state:
    st.info("Click **Train models** in the sidebar after uploading the dataset.")
    st.stop()

results_df = st.session_state["fraud_results"]
st.subheader("3. Model performance")
st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
        "F1-score": "{:.4f}", "ROC-AUC": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)
st.caption(
    "Because fraud is rare, compare precision, recall, F1-score, and ROC-AUC—not accuracy alone. "
    "A higher recall catches more fraud cases but may also flag more legitimate transactions."
)

chart_df = results_df.set_index("Model")[["Precision", "Recall", "F1-score", "ROC-AUC"]]
st.bar_chart(chart_df)

st.subheader("4. Confusion matrix")
selected_model = st.selectbox("Choose model", list(st.session_state["fraud_models"].keys()))
y_test = st.session_state["fraud_y_test"]
y_pred = st.session_state["fraud_predictions"][selected_model]
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

fig, ax = plt.subplots()
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate (0)", "Fraud (1)"]
).plot(ax=ax, values_format="d", colorbar=False)
plt.xticks(rotation=15)
st.pyplot(fig)
plt.close(fig)

st.subheader("5. Test a transaction from the held-out test set")
st.write(
    "Choose a test-set row to see the selected model's prediction. The probability is a model score, "
    "not a guarantee or a calibrated real-world risk estimate."
)
X_test = st.session_state["fraud_X_test"]
row_position = st.number_input(
    "Test-set row number",
    min_value=0,
    max_value=max(0, len(X_test) - 1),
    value=0,
    step=1
)
if st.button("Predict selected transaction"):
    model = st.session_state["fraud_models"][selected_model]
    row = X_test.iloc[[int(row_position)]]
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])

    c1, c2 = st.columns(2)
    c1.metric("Predicted fraud probability", f"{probability:.2%}")
    c2.metric("Predicted class", "Potential fraud (1)" if prediction == 1 else "Likely legitimate (0)")
    if prediction == 1:
        st.error("The model flags this transaction as potentially fraudulent. Review it; do not treat this as proof.")
    else:
        st.success("The model classifies this transaction as likely legitimate. False negatives are still possible.")
    with st.expander("View transaction features"):
        st.dataframe(row, use_container_width=True)

st.subheader("6. Download model comparison")
st.download_button(
    "Download metrics as CSV",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="fraud_model_comparison.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("Project for learning and demonstration. Validate carefully before any real-world deployment.")
