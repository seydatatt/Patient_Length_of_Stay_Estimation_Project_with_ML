# 🏥 Patient Length of Stay Prediction Project 🏥

## 🚀 Project Goal
Predict how long a patient will stay in the hospital using **Machine Learning**. This helps hospitals plan better and improve patient care!

---

## 📊 Dataset
We used the **Hospital Inpatient Discharges (SPARCS De-Identified) 2021** dataset from Kaggle:

👉 [Dataset Link](https://www.kaggle.com/datasets/bhautikmangukiya12/hospital-inpatient-discharges-dataset)

---

## 🧹 Data Cleaning & Preparation
- Replaced `"120 +"` with number `120` in Length of Stay (LOS).
- Converted categories like Age Group, Gender, Race, etc. into numbers using encoding.
- Removed patients who **expired** during stay.
- Dropped unnecessary columns for simplicity.
- Removed rows with missing important values.
- Split data into **training (80%)** and **testing (20%)** sets.

---

## 🔍 Exploratory Data Analysis (EDA)
We visualized data to understand it better using graphs:
- 💳 Payment Type vs Length of Stay  
- 👶 Age groups of Medicare patients  
- 🏥 Type of Admission vs Length of Stay  
- 🎂 Age Group vs Length of Stay  

---

## 🤖 Modeling

### 1️⃣ Regression Model  
- Used a **Decision Tree Regressor** to predict exact days of stay.  
- Controlled complexity with `max_depth=10` to avoid overfitting.  
- Measured accuracy with RMSE (Root Mean Squared Error).

### 2️⃣ Classification Model  
- Grouped Length of Stay into categories (bins):  
  `[0-5, 6-10, 11-20, 21-30, 31-50, 51-120+]`  
- Used **Decision Tree Classifier** to predict which bin the stay falls into.  
- Checked model accuracy and generated classification reports.  

---

## 📈 Results
- Regression RMSE scores for training and testing data printed.
- Classification accuracy scores shown for train & test sets.
- Confusion matrix visualized to see prediction quality.

---

## 💾 Saving the Model
The trained classification model is saved for future use with `pickle`:

```python
with open("patient_los_model.pkl", "wb") as f:
    pickle.dump(dtree, f)
```

---

## 🛠️ How to Run
1. Download dataset and place it in the project folder.  
2. Install dependencies:
   ```
   pip install numpy pandas scikit-learn seaborn matplotlib
   ```
3. Run the Python script.  
4. Check output, plots, and saved model file.

---

## 🔮 Future Ideas
- Try other models like Random Forest, Gradient Boosting, or Neural Networks.  
- Tune model parameters for better accuracy.  
- Handle missing data with smart imputations instead of dropping rows.  
- Add feature importance plots.  
- Build a user-friendly app (e.g., with Streamlit) to predict LOS interactively.

---
## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Thank you for checking out this project! 😊
