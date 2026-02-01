
# Student Academic Performance Prediction using Feature Selection and Multi-Model Machine Learning

This project predicts student academic performance (grade categories) using machine learning techniques and feature selection.

## 📊 Dataset
Kaggle Dataset: *Student Success: Factors & Insights*  
Place the dataset CSV inside the `data/` folder and rename it to:

`student_performance.csv`

## 🚀 Project Structure

```
student-performance-ml-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── student_performance.csv
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── feature_selection.py
│   └── main.py
│
├── results/
└── report/
    └── Final_Report.pdf
```

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/main.py
```

## 🧠 Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (Best Performer)
- K-Nearest Neighbors

## 🏆 Key Findings
Top predictive features:
Attendance, Hours Studied, Previous Scores, Tutoring Sessions, Sleep Hours,
Parental Involvement, Access to Resources, Motivation Level, Family Income, Peer Influence.

SVM achieved the highest performance with full features.

---
Author: Joud Alsuhaibany
