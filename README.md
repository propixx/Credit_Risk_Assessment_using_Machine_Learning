# 📊 Credit Risk Assessment using Machine Learning

An end-to-end credit risk prediction model leveraging machine learning techniques to assess the likelihood of default for credit applicants based on financial and demographic profiles. This project uses the German Credit dataset and provides a structured approach from data preprocessing to model evaluation.

---

## 🚀 Project Overview

Credit risk assessment is a critical task for financial institutions to minimize potential losses. In this project, we developed classification models to predict whether a loan applicant poses a low or high risk based on historical credit data.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy, Matplotlib** – Data Handling and Visualization
- **Scikit-learn** – Machine Learning Models
- **Streamlit** – Interactive Web App (Optional Future Scope)

---

## 📂 Project Structure

```
credit-risk-assessment/
├── Credit_risk_notebook.ipynb
├── Features_Target_Description.xlsx
├── case_study1.xlsx
├── case_study2.xlsx
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📈 Dataset Description

- **Source:** German Credit Dataset  
- **Size:** ~1,000 records
- **Features:** 20 financial and demographic variables
- **Target:** Credit Risk (Good / Bad)

---

## ⚙️ Key Features

- ✅ Data Cleaning and Preprocessing
- ✅ Feature Description and Selection
- ✅ Statistical Analysis and Visualization
- ✅ Model Building using Logistic Regression and Random Forest
- ✅ Model Evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- ✅ Real-time Credit Risk Prediction via user-input (Optional: Streamlit App)

---

## 🏗️ Steps Performed

1. **Data Cleaning:**
   - Removed duplicates, handled missing values.
   - Converted categorical variables using encoding techniques.

2. **Exploratory Data Analysis:**
   - Performed statistical summaries.
   - Visualized class imbalance and key feature distributions.

3. **Feature Selection:**
   - Used Variance Inflation Factor (VIF) to detect multicollinearity.
   - Applied Chi-Square tests to assess feature relevance.

4. **Model Building:**
   - Trained Logistic Regression and Random Forest classifiers.
   - Evaluated models using cross-validation and ROC-AUC scores.

5. **Result Interpretation:**
   - Generated confusion matrix and classification reports.
   - Provided actionable insights for high-risk applicant identification.

---

## 📦 Installation

```bash
git clone https://github.com/propixx/Credit_Risk_Assessment_using_Machine_Learning.git
cd credit-risk-assessment
pip install -r requirements.txt
```

---

## ▶️ Running the Notebook

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```
2. Open `Credit_risk_notebook.ipynb` and execute the cells step-by-step.

---

## 🎯 Future Scope

- Deploy a **Streamlit Web App** for live credit risk predictions.
- Add additional datasets for more robust training.
- Integrate advanced ML models like XGBoost or LightGBM for better performance.

---

## 🔗 Resources

- [German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## ✨ Acknowledgments

- This project was inspired by publicly available datasets and machine learning practices.
- Referenced the work of CampusX and other community-driven tutorials.

---
