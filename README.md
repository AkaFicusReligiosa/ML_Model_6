
# Loan Status Predictor using Support Vector Classifier (SVC)

This project implements a **Support Vector Machine (SVC)** model to predict loan approval status based on applicant details. The system helps financial institutions make data-driven decisions about whether to approve or reject loan applications.

---

## 🎯 Objective

To build a machine learning classification model using **SVC** that accurately predicts whether a loan application will be **approved (Y)** or **rejected (N)** based on applicant information.

---

## 📁 Dataset

- **Source**: [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Target Variable**: `Loan_Status` (`Y` = Approved, `N` = Not Approved)
- **Features Include**:
  - Gender
  - Married
  - Dependents
  - Education
  - Self_Employed
  - ApplicantIncome
  - CoapplicantIncome
  - LoanAmount
  - Loan_Amount_Term
  - Credit_History
  - Property_Area

---

## 🚀 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Jupyter Notebook

---

## 🧠 Model Workflow

1. **Load and Explore Data**
2. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Feature scaling
3. **Train/Test Split**
4. **Model Training**
   - Support Vector Classifier (SVC) with hyperparameter tuning
5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1 Score
6. **Prediction on New Applications**

---

## 📊 Evaluation Metrics

- **Confusion Matrix**
- **Accuracy Score**
- **Precision / Recall / F1-score**
- **Cross-validation (optional)**

---

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/loan-status-predictor-svc.git
   cd loan-status-predictor-svc
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter notebook loan_status_predictor.ipynb
   ```

---

## 🔮 Sample Prediction Code

```python
sample_input = [[1, 0, 1, 0, 1, 4583, 1508, 128.0, 360.0, 1.0, 2]]  # Example input after encoding
prediction = svc_model.predict(sample_input)
print("Loan Status Prediction:", "Approved" if prediction[0] == 'Y' else "Not Approved")
```

---

## 📈 Results

* The model achieved **\~80-85% accuracy** on the test data after preprocessing and tuning.
* SVC showed strong performance especially when the data was properly scaled and encoded.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

* [Kaggle: Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
* Scikit-learn
* Python open-source community

```
