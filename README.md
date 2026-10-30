**App Link:** https://financial-fraud-detection-imdalkweuazo4fzrypr8p6.streamlit.app/   


# 🛡️ AI-Powered Financial Fraud Detection System

### Machine Learning | Artificial Neural Networks | Python | Streamlit

## 📌 About the Project

The **AI-Powered Financial Fraud Detection System** is a machine learning project designed to identify potentially fraudulent credit card transactions using Artificial Intelligence (AI) and data analysis.

The project implements and compares three classification algorithms: Logistic Regression, Random Forest, and Artificial Neural Network (ANN). It provides an interactive Streamlit dashboard where users can upload transaction datasets, train models, evaluate performance, visualize results, and test transactions for potential fraud.

## 🎯 Project Objectives

- Detect potentially fraudulent financial transactions.
- Apply Machine Learning and Artificial Neural Networks to transaction data.
- Compare the performance of multiple classification models.
- Visualize fraud detection results using charts and confusion matrices.
- Develop an interactive web application for fraud analysis.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning and model evaluation |
| Artificial Neural Network (MLP) | Learning complex transaction patterns |
| Matplotlib | Data visualization |
| Streamlit | Interactive web dashboard |

## 🤖 Machine Learning Algorithms

1. **Logistic Regression** — Baseline classification model.
2. **Random Forest Classifier** — Ensemble learning algorithm for classification.
3. **Artificial Neural Network (ANN)** — Neural network model for learning patterns in transaction data.

## ✨ Key Features

- Upload transaction datasets in CSV format.
- Preview transaction data and analyze fraud distribution.
- Train and compare three machine learning models.
- Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Display confusion matrices and performance charts.
- Predict potential fraud for selected test transactions.
- Download model comparison results as a CSV file.

## 📊 Dataset

This project uses the publicly available **Credit Card Fraud Detection Dataset** from Kaggle.

**Dataset link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains approximately 284,807 transactions, including 492 fraudulent transactions.

The `Class` column is the target variable:

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

Most transaction features (`V1` to `V28`) are anonymized numerical variables.

## 📂 Project Structure

```text
financial-fraud-detection/
│
├── app.py
├── requirements.txt
└── README.md
```

- `app.py` — Main Streamlit application containing the model training, evaluation, and prediction code.
- `requirements.txt` — Required Python libraries.
- `README.md` — Project documentation.

The transaction dataset is uploaded through the application and does not need to be stored in the repository.

## ⚙️ Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/financial-fraud-detection.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Open the project folder

```bash
cd financial-fraud-detection
```

### Step 3: Install the dependencies

Make sure Python is installed, then run:

```bash
python -m pip install -r requirements.txt
```

### Step 4: Download the dataset

Download `creditcard.csv` from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Step 5: Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser. Upload the dataset using the sidebar and click **Train models**.

## 📈 Model Evaluation

The models are evaluated using the following metrics:

- **Accuracy:** Overall proportion of correctly classified transactions.
- **Precision:** Proportion of predicted fraud cases that are actually fraud.
- **Recall:** Proportion of actual fraud cases detected by the model.
- **F1-score:** Balance between precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between the two classes across different thresholds.

Because fraudulent transactions are rare, precision, recall, F1-score, and ROC-AUC are particularly important. Accuracy alone can be misleading.

## 🌐 Deployment

The application can be deployed using Streamlit Community Cloud.

1. Upload `app.py`, `requirements.txt`, and `README.md` to GitHub.
2. Visit https://share.streamlit.io/
3. Connect your GitHub repository.
4. Select `app.py` as the main application file.
5. Deploy the application.

## 🎓 Learning Outcomes

This project demonstrates practical skills in:

- Data preprocessing and exploratory data analysis.
- Supervised machine learning.
- Artificial Neural Networks.
- Imbalanced classification.
- Model evaluation and comparison.
- Data visualization and dashboard development.
- Deploying a Python web application.

## ⚠️ Disclaimer

This project is developed for educational and demonstration purposes only. Model predictions do not prove that a transaction is fraudulent. The system has not been validated for real-world financial decisions.

## 👨‍💻 Author

Tripti Saxena
