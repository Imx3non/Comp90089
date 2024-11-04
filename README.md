# Comp90089 GROUP 28
# Machine Learning-Based Stroke Risk Prediction in Hypertensive Patients

This project applies machine learning to predict stroke risk among hypertensive patients by identifying key predictive factors from the MIMIC-IV dataset. It uses logistic regression, random forest, XGBoost, and SVM models, with a focus on interpretable and accurate predictions to assist in clinical risk assessments.

## Project Structure

- **`LR&RF - final.ipynb`**: Contains baseline models—Logistic Regression and Random Forest—for initial stroke prediction.
- **`Herro's code/`**: Code for data processing and the XGBoost model implementation.
- **`svm_classifier/`**: SVM model for stroke prediction.
- **`final_code.ipynb`**: Integrates all models and outputs final predictions with combined feature sets for improved accuracy and interpretability.

## Dataset

- **MIMIC-IV Dataset**: A publicly available healthcare dataset.

## Methodology

1. **Data Preprocessing**:
   - Label encoding, feature cleaning, and class balancing (using SMOTE).
  
2. **Feature Selection**:
   - Based on feature importance from Random Forest and XGBoost, the top predictors were selected, including factors such as age, anemia, and atrial fibrillation.

3. **Model Training**:
   - Baseline models: Logistic Regression and Random Forest.
   - Additional models: XGBoost and SVM.
   - Models evaluated using metrics like accuracy, precision, recall, F1 score, and AUC.

4. **Final Model Integration**:
   - Combined insights from individual models to create an ensemble approach in `final_code.ipynb`.


## Getting Started

1. **Prerequisites**:
   - Python 3.8 or above

2. **Usage**:
   - Run `LR&RF - final.ipynb` for baseline model results.
   - Use `Herro's code/` for XGBoost-related data processing and model training.
   - Check `svm_classifier/` for the SVM implementation.
   - Finally, execute `final_code.ipynb` to see the integrated model results.

## Contributions

- **Wanyi Lu**: Data Curation, Formal Analysis, Methodology, Logistic Regression and Random Forest Algorithm Code Implementation, Methods Writing – Original Draft, Writing – Review and Editing
- **Mingjun Xie**: Data Curation, Formal Analysis, XGboost Algorithm Code Implementation, Integration Result Code Implementation, Introduction and Conclusion Writing – Original Draft, Writing – Review and Editing
- **Rajneesh Gokool**: Conceptualization, Formal Analysis, Methodology, SVM Algorithm Code Implementation, Results and Discussion Writing – Original Draft, Writing – Review and Editing

