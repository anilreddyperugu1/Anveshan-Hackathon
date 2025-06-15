# 🕵️ Fake Job Postings Detection using Machine Learning

This project builds a **binary classification model** to identify potentially fraudulent job postings using **natural language processing (NLP)** and **machine learning**. It was developed as part of a hackathon challenge.

---

## 📚 Index

1. [Project Overview](#project-overview)  
2. [Key Features & Technologies Used](#key-features--technologies-used)  
3. [Folder Structure](#folder-structure)  
4. [Setup Instructions](#setup-instructions)  
5. [Modeling Workflow](#modeling-workflow)  
6. [Results & Evaluation](#results--evaluation)

---

## 📌 Project Overview

Job portals often face the issue of fake or scam job postings that waste applicant time and can cause harm. This project applies machine learning to detect such **fraudulent listings** by analyzing job titles, descriptions, and metadata.

We trained and evaluated multiple models and selected the **XGBoost classifier**, achieving an **F1-score of 0.825** on the validation set.

---

## 🚀 Key Features & Technologies Used

- **Python**
- **scikit-learn** for ML pipeline
- **XGBoost** for binary classification
- **TF-IDF Vectorization** for text features
- **Imbalanced Data Handling** using resampling
- **Threshold tuning** to optimize F1-score
- **Pandas** for data handling
- **Matplotlib (optional)** for visualization

---

## 🗂 Folder Structure

| -- Datasets 
|        |-- test data
|        |-- train data
|--Models
|     |-- xgboost model
|     | -- xgboost vectorizer model
|-- testingfile
|-- trainingfile


---

## ⚙️ Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/fake-job-detection.git
   cd fake-job-detection

2.**Install dependencies**
   * pip install -r requirements.txt

3. **Run training pipeline**
   * Open notebooks/training_pipeline.ipynb
   * Follow cells to train the model, save vectorizer and classifier
4. **Run test prediction**
   * Open notebooks/test_prediction.ipynb
   * Loads model + vectorizer, runs predictions on test data
   * Saves submission file to submission/submission.csv

**Modeling Workflow**
✔️ **Data Preprocessing**
   * Combined title and description
   * Cleaned text (lowercasing, optional punctuation/stemming)
   * TF-IDF vectorization

✔️ **Resampling for Class Imbalance**
   * Applied random oversampling to balance fake/real jobs
   * Evaluated class distribution after resampling

✔️ **Model Training**
   * Used XGBoostClassifier with:
   * use_label_encoder=False
   * eval_metric='logloss'
   * random_state=42
   * Trained on balanced TF-IDF features
   * Selected threshold that maximized F1-score

✔️ **Evaluation**
**Achieved:**
   **F1-score:** 0.825 on validation set
   **Training Accuracy:** 1.0
   **Validation Accuracy:** 0.98

📈 **Results & Submission**
Final test predictions saved to submission/submission.csv

**Prediction class distribution:**

**0 (real jobs):** 3502
**1 (fraud jobs):** 74

Model used: XGBoost with TF-IDF features

🧠 **Author Notes**
This project was developed as part of a machine learning hackathon. The goal was to demonstrate end-to-end modeling from text cleaning to model deployment.

📫 Feel free to fork or contribute!

### ✅ `requirements.txt` example:
pandas
numpy
scikit-learn
xgboost
matplotlib
classification report
vectorizer

