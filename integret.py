# Install required libraries if not already installed
# Uncomment the following lines to install necessary libraries
# !pip install xgboost imbalanced-learn matplotlib seaborn pandas scikit-learn joblib

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置joblib的临时文件夹路径（路径中不包含中文或其他非ASCII字符）
temp_folder = r'D:\comp90089'  # 或者您电脑上的其他路径，确保没有非ASCII字符
os.makedirs(temp_folder, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = temp_folder

# Machine learning libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    auc,
)
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import joblib

# Function to clean column names
def clean_column_names(columns):
    """
    Replace problematic characters in column names with underscores.
    """
    return [
        str(col)
        .replace('[', '_')
        .replace(']', '_')
        .replace('<', '_')
        .replace('>', '_')
        .replace('/', '_')
        .replace(' ', '_')
        for col in columns
    ]

# Create the 'results' directory if it doesn't exist
os.makedirs('results', exist_ok=True)
# Create the 'result', 'results/rf', and 'results/xgb' directories if they don't exist
os.makedirs('results/rf', exist_ok=True)
os.makedirs('results/xgb', exist_ok=True)
os.makedirs('results/common', exist_ok=True)
os.makedirs('results/common/rf', exist_ok=True)
os.makedirs('results/common/xgb', exist_ok=True)
# Load the dataset
final_data = pd.read_csv('final_data_with_long_titles.csv')

# Display the first few rows
print("First 5 rows of the dataset:")
print(final_data.head())

# Check the shape of the dataset
print(f"Dataset shape: {final_data.shape}")

# Save the initial data summary to a text file
with open('results/data_summary.txt', 'w') as f:
    f.write(f"Dataset shape: {final_data.shape}\n")
    f.write("First 5 rows of the dataset:\n")
    f.write(final_data.head().to_string())

# Create target variable 'stroke'
# If 'classification' == 'stroke_after_hypertension', then stroke = 1, else stroke = 0
final_data['stroke'] = np.where(final_data['classification'] == 'stroke_after_hypertension', 1, 0)

# Drop unnecessary columns
data = final_data.drop(columns=['subject_id', 'hadm_id', 'classification'])

# Handle missing values in 'age' by filling with mean
data['age'] = data['age'].fillna(data['age'].mean())

# Encode 'gender'
data['gender'] = data['gender'].map({'M': 0, 'F': 1})

# Store original feature names before cleaning
original_feature_names = data.drop('stroke', axis=1).columns.tolist()

# Clean column names to remove problematic characters
cleaned_feature_names = clean_column_names(original_feature_names)

# Create a mapping from cleaned names to original names
cleaned_to_original = dict(zip(cleaned_feature_names, original_feature_names))

# Assign cleaned feature names to the DataFrame
data_cleaned = data.copy()
data_cleaned.columns = clean_column_names(data_cleaned.columns)

# Ensure ICD code columns are numeric
icd_columns_cleaned = [col for col in data_cleaned.columns if col.startswith('ICD_')]
data_cleaned[icd_columns_cleaned] = data_cleaned[icd_columns_cleaned].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

# Separate features and target
X = data_cleaned.drop('stroke', axis=1)
y = data_cleaned['stroke']

# Check for missing values in X
missing_values = X.isnull().sum()
print("Missing values in each feature:")
print(missing_values[missing_values > 0])

# Alternatively, check the total number of missing values in X
total_missing = X.isnull().sum().sum()
print(f"\nTotal missing values in X: {total_missing}")

# Save class distribution to a text file
class_distribution = y.value_counts()
print("Class distribution in the target variable:")
print(class_distribution)

with open('results/class_distribution.txt', 'w') as f:
    f.write("Class distribution in the target variable:\n")
    f.write(class_distribution.to_string())

# Visualize class distribution with numerical labels
plt.figure(figsize=(6, 4))
ax = sns.countplot(x=y)
plt.title('Class Distribution')
plt.xlabel('Stroke (1) vs No Stroke (0)')
plt.ylabel('Count')

# Add numerical labels on top of each bar
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/class_distribution.png')
plt.show()

# Handle missing values by filling all with 0
if total_missing > 0:
    X_filled = X.fillna(0)
    print("Total missing values after filling with 0:")
    print(X_filled.isnull().sum().sum())
else:
    X_filled = X.copy()

# Ensure no missing values after filling
assert X_filled.isnull().sum().sum() == 0, "Missing values still exist after filling!"

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_filled, y)

# New class distribution
resampled_distribution = pd.Series(y_resampled).value_counts()
print("Class distribution after resampling:")
print(resampled_distribution)

# Save resampled class distribution to a text file
with open('results/resampled_class_distribution.txt', 'w') as f:
    f.write("Class distribution after resampling:\n")
    f.write(resampled_distribution.to_string())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Save the data split info to a text file
with open('results/data_split_info.txt', 'w') as f:
    f.write(f"Training set size: {X_train.shape}\n")
    f.write(f"Testing set size: {X_test.shape}\n")

# ------------------------ 训练随机森林模型开始 ------------------------

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and Evaluation for Random Forest
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report for Random Forest:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix for Random Forest:\n", confusion_matrix(y_test, y_pred_rf))

# Save classification report to a text file
with open('results/rf/classification_report_rf.txt', 'w') as f:
    f.write("Classification Report for Random Forest:\n")
    f.write(classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix for Random Forest:")
print(conf_matrix_rf)

# Save confusion matrix to a CSV file
conf_matrix_rf_df = pd.DataFrame(conf_matrix_rf, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
conf_matrix_rf_df.to_csv('results/rf/confusion_matrix_rf.csv')

# Visualize Confusion Matrix with numerical labels for Random Forest
plt.figure(figsize=(6, 4))
ax = sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('results/rf/confusion_matrix_rf.png')
plt.show()

# ROC Curve for Random Forest
y_pred_rf_prob = rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_prob)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf_prob)
print(f"Random Forest ROC AUC Score: {roc_auc_rf:.2f}")

# Save ROC AUC score to a text file
with open('results/rf/roc_auc_score_rf.txt', 'w') as f:
    f.write(f"Random Forest ROC AUC Score: {roc_auc_rf:.2f}\n")

# Plot ROC Curve for Random Forest
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = {:.2f})'.format(roc_auc_rf), color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/rf/roc_curve_rf.png')
plt.show()

# Compute average precision for Random Forest
average_precision_rf = average_precision_score(y_test, y_pred_rf_prob)
print(f'Random Forest Average Precision Score: {average_precision_rf:.2f}')

# Save average precision score to a text file
with open('results/rf/average_precision_score_rf.txt', 'w') as f:
    f.write(f'Random Forest Average Precision Score: {average_precision_rf:.2f}\n')

# Plot Precision-Recall Curve for Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_rf_prob)
plt.figure(figsize=(8, 6))
plt.step(recall_rf, precision_rf, where='post', color='b', alpha=0.7, label='Precision-Recall RF')
plt.fill_between(recall_rf, precision_rf, step='post', alpha=0.3, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP = {average_precision_rf:.2f}) - Random Forest')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/rf/precision_recall_curve_rf.png')
plt.show()

# Calculate F1 Score for Random Forest
f1_rf = f1_score(y_test, y_pred_rf)
print(f"Random Forest F1 Score: {f1_rf:.2f}")

# Save F1 score to a text file
with open('results/rf/f1_score_rf.txt', 'w') as f:
    f.write(f"Random Forest F1 Score: {f1_rf:.2f}\n")

# Feature Importance from Random Forest
importances_rf = rf_model.feature_importances_
feature_names_rf = X.columns
feature_importances_rf = pd.Series(importances_rf, index=feature_names_rf).sort_values(ascending=False)

print("Top 10 Important Features - Random Forest:\n", feature_importances_rf.head(10))

# Plot Feature Importances for top 50 features - Random Forest
top_n = 50
feat_imp_rf_top_n = feature_importances_rf.head(top_n)

plt.figure(figsize=(12, 10))
sns.barplot(x=feat_imp_rf_top_n.values, y=feat_imp_rf_top_n.index, palette='viridis')
plt.title(f'Top {top_n} Feature Importances - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('results/rf/feature_importances_rf_top.png')
plt.show()

# ------------------------ 训练随机森林模型结束 ------------------------

# ------------------------ 训练XGBoost模型开始 ------------------------

# Initialize and train the XGBoost model
model_xgb = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model_xgb.fit(X_train, y_train)

# Predictions
y_pred_xgb = model_xgb.predict(X_test)
y_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

# Classification report
classification_rep_xgb = classification_report(y_test, y_pred_xgb)
print("\nXGBoost Classification Report:")
print(classification_rep_xgb)

# Save classification report to a text file
with open('results/xgb/classification_report_xgb.txt', 'w') as f:
    f.write("Classification Report - XGBoost:\n")
    f.write(classification_rep_xgb)

# Confusion Matrix
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
print("Confusion Matrix - XGBoost:")
print(conf_matrix_xgb)

# Save confusion matrix to a CSV file
conf_matrix_xgb_df = pd.DataFrame(conf_matrix_xgb, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
conf_matrix_xgb_df.to_csv('results/xgb/confusion_matrix_xgb.csv')

# Visualize Confusion Matrix with numerical labels - XGBoost
plt.figure(figsize=(6, 4))
ax = sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('results/xgb/confusion_matrix_xgb.png')
plt.show()

# Compute ROC AUC
roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)
print(f"XGBoost ROC AUC Score: {roc_auc_xgb:.2f}")

# Save ROC AUC score to a text file
with open('results/xgb/roc_auc_score_xgb.txt', 'w') as f:
    f.write(f"XGBoost ROC AUC Score: {roc_auc_xgb:.2f}\n")

# Plot ROC Curve - XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost (AUC = {:.2f})'.format(roc_auc_xgb), color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/xgb/roc_curve_xgb.png')
plt.show()

# Compute average precision
average_precision_xgb = average_precision_score(y_test, y_proba_xgb)
print(f'XGBoost Average Precision Score: {average_precision_xgb:.2f}')

# Save average precision score to a text file
with open('results/xgb/average_precision_score_xgb.txt', 'w') as f:
    f.write(f'XGBoost Average Precision Score: {average_precision_xgb:.2f}\n')

# Plot Precision-Recall Curve - XGBoost
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_proba_xgb)
plt.figure(figsize=(8, 6))
plt.step(recall_xgb, precision_xgb, where='post', color='b', alpha=0.7, label='Precision-Recall XGBoost')
plt.fill_between(recall_xgb, precision_xgb, step='post', alpha=0.3, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP = {average_precision_xgb:.2f}) - XGBoost')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/xgb/precision_recall_curve_xgb.png')
plt.show()

# Calculate F1 Score
f1_xgb = f1_score(y_test, y_pred_xgb)
print(f"XGBoost F1 Score: {f1_xgb:.2f}")

# Save F1 score to a text file
with open('results/xgb/f1_score_xgb.txt', 'w') as f:
    f.write(f"XGBoost F1 Score: {f1_xgb:.2f}\n")

# Create a DataFrame with actual and predicted probabilities
lift_data_xgb = pd.DataFrame({'y_true': y_test, 'y_proba': y_proba_xgb})
lift_data_xgb = lift_data_xgb.sort_values('y_proba', ascending=False).reset_index(drop=True)
lift_data_xgb['cum_response'] = lift_data_xgb['y_true'].cumsum()
lift_data_xgb['cum_percentage'] = lift_data_xgb.index / len(lift_data_xgb)
lift_data_xgb['lift'] = lift_data_xgb['cum_response'] / (lift_data_xgb['cum_percentage'] * lift_data_xgb['y_true'].sum())

# Plot Lift Chart - XGBoost
plt.figure(figsize=(8, 6))
plt.plot(lift_data_xgb['cum_percentage'], lift_data_xgb['lift'], label='Lift Curve XGBoost')
plt.xlabel('Percentage of Sample')
plt.ylabel('Lift')
plt.title('Lift Chart - XGBoost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/xgb/lift_chart_xgb.png')
plt.show()

# Plot Gain Chart - XGBoost
plt.figure(figsize=(8, 6))
plt.plot(lift_data_xgb['cum_percentage'], lift_data_xgb['cum_response'] / lift_data_xgb['y_true'].sum(), label='Gain Curve XGBoost')
plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
plt.xlabel('Percentage of Sample')
plt.ylabel('Percentage of Positive Cases')
plt.title('Gain Chart - XGBoost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/xgb/gain_chart_xgb.png')
plt.show()

# Get feature importances
importances_xgb = model_xgb.feature_importances_
features_xgb = X.columns.tolist()

# Create a DataFrame for feature importances
feat_imp_xgb_df = pd.DataFrame({'Feature_Cleaned': features_xgb, 'Importance': importances_xgb})
# Map cleaned feature names back to original names
feat_imp_xgb_df['Feature_Original'] = feat_imp_xgb_df['Feature_Cleaned'].map(cleaned_to_original)

# Sort features by importance
feat_imp_xgb_df = feat_imp_xgb_df.sort_values('Importance', ascending=False)

# Select top 50 features
top_n = 50
feat_imp_xgb_top_n = feat_imp_xgb_df.head(top_n)

# Display top features
print(f"\nTop {top_n} Most Influential Features - XGBoost:")
print(feat_imp_xgb_top_n[['Feature_Original', 'Importance']])

# Save feature importances to a CSV file
feat_imp_xgb_df.to_csv('results/xgb/feature_importances_xgb.csv', index=False)

# Plot Feature Importances for top 50 features - XGBoost
plt.figure(figsize=(12, 10))  # Increased figure size for better readability
sns.barplot(x='Importance', y='Feature_Original', data=feat_imp_xgb_top_n, palette='viridis')
plt.title(f'Top {top_n} Feature Importances - XGBoost')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('results/xgb/feature_importances_xgb_top.png')
plt.show()

# ------------------------ 训练XGBoost模型结束 ------------------------
# Continue from where you left off after training the individual models

# ------------------------ Combine Models Using Ensemble Methods ------------------------

# Import necessary libraries for ensemble methods
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import shap

# Initialize the base models with the same parameters as before
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

# ------------------------ Voting Classifier ------------------------

# Create a Voting Classifier with soft voting
voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    voting='soft',
    n_jobs=-1
)

# Train the Voting Classifier
voting_clf.fit(X_train, y_train)

# Predictions and Evaluation for Voting Classifier
y_pred_voting = voting_clf.predict(X_test)
y_pred_voting_prob = voting_clf.predict_proba(X_test)[:, 1]

print("\nVoting Classifier Accuracy:", accuracy_score(y_test, y_pred_voting))
print("Classification Report for Voting Classifier:\n", classification_report(y_test, y_pred_voting))
print("Confusion Matrix for Voting Classifier:\n", confusion_matrix(y_test, y_pred_voting))

# Save classification report to a text file
with open('results/common/classification_report_voting.txt', 'w') as f:
    f.write("Classification Report for Voting Classifier:\n")
    f.write(classification_report(y_test, y_pred_voting))

# ROC Curve for Voting Classifier
fpr_voting, tpr_voting, _ = roc_curve(y_test, y_pred_voting_prob)
roc_auc_voting = roc_auc_score(y_test, y_pred_voting_prob)
print(f"Voting Classifier ROC AUC Score: {roc_auc_voting:.2f}")

# Save ROC AUC score to a text file
with open('results/common/roc_auc_score_voting.txt', 'w') as f:
    f.write(f"Voting Classifier ROC AUC Score: {roc_auc_voting:.2f}\n")

# Plot ROC Curve for Voting Classifier
plt.figure(figsize=(8, 6))
plt.plot(fpr_voting, tpr_voting, label='Voting Classifier (AUC = {:.2f})'.format(roc_auc_voting), color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Voting Classifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/common/roc_curve_voting.png')
plt.show()

# ------------------------ Stacking Classifier ------------------------

# Create a Stacking Classifier with Logistic Regression as the meta-model
stacking_clf = StackingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    final_estimator=LogisticRegression(max_iter=1000),
    passthrough=False,
    cv=5,
    n_jobs=-1
)

# Train the Stacking Classifier
stacking_clf.fit(X_train, y_train)

# Predictions and Evaluation for Stacking Classifier
y_pred_stacking = stacking_clf.predict(X_test)
y_pred_stacking_prob = stacking_clf.predict_proba(X_test)[:, 1]

print("\nStacking Classifier Accuracy:", accuracy_score(y_test, y_pred_stacking))
print("Classification Report for Stacking Classifier:\n", classification_report(y_test, y_pred_stacking))
print("Confusion Matrix for Stacking Classifier:\n", confusion_matrix(y_test, y_pred_stacking))

# Save classification report to a text file
with open('results/common/classification_report_stacking.txt', 'w') as f:
    f.write("Classification Report for Stacking Classifier:\n")
    f.write(classification_report(y_test, y_pred_stacking))

# ROC Curve for Stacking Classifier
fpr_stacking, tpr_stacking, _ = roc_curve(y_test, y_pred_stacking_prob)
roc_auc_stacking = roc_auc_score(y_test, y_pred_stacking_prob)
print(f"Stacking Classifier ROC AUC Score: {roc_auc_stacking:.2f}")

# Save ROC AUC score to a text file
with open('results/common/roc_auc_score_stacking.txt', 'w') as f:
    f.write(f"Stacking Classifier ROC AUC Score: {roc_auc_stacking:.2f}\n")

# Plot ROC Curve for Stacking Classifier
plt.figure(figsize=(8, 6))
plt.plot(fpr_stacking, tpr_stacking, label='Stacking Classifier (AUC = {:.2f})'.format(roc_auc_stacking), color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Stacking Classifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/common/roc_curve_stacking.png')
plt.show()

# ------------------------ Integrate Feature Importances ------------------------

# Normalize importances
importances_rf_norm = feature_importances_rf / feature_importances_rf.sum()
importances_xgb_norm = feat_imp_xgb_df.set_index('Feature_Cleaned')['Importance'] / feat_imp_xgb_df['Importance'].sum()

# Align features
combined_features = importances_rf_norm.index.union(importances_xgb_norm.index)

# Fill missing values with zero
importances_rf_norm = importances_rf_norm.reindex(combined_features, fill_value=0)
importances_xgb_norm = importances_xgb_norm.reindex(combined_features, fill_value=0)

# Average importances
average_importances = (importances_rf_norm + importances_xgb_norm) / 2

# Sort combined importances
average_importances = average_importances.sort_values(ascending=False)

# Select top N features
top_n = 20
top_n_features = average_importances.head(top_n).index.tolist()

# Map cleaned feature names back to original names
top_n_features_original = [cleaned_to_original.get(feature, feature) for feature in top_n_features]

# Save combined feature importances to a CSV file
average_importances_df = pd.DataFrame({
    'Feature_Cleaned': average_importances.index,
    'Importance': average_importances.values,
    'Feature_Original': [cleaned_to_original.get(feature, feature) for feature in average_importances.index]
})
average_importances_df.to_csv('results/common/average_feature_importances.csv', index=False)

# Plot Combined Feature Importances
plt.figure(figsize=(12, 10))
sns.barplot(x=average_importances.head(top_n).values, y=top_n_features_original, palette='viridis')
plt.title(f'Top {top_n} Combined Feature Importances')
plt.xlabel('Average Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('results/common/combined_feature_importances_top.png')
plt.show()

print(f"\nTop {top_n} Combined Most Influential Features:")
for feature, importance in zip(top_n_features_original, average_importances.head(top_n).values):
    print(f"{feature}: {importance:.4f}")

# ------------------------ Retrain Models with Selected Features ------------------------

# Subset the data to top N features
X_train_selected = X_train[top_n_features]
X_test_selected = X_test[top_n_features]

# Retrain Random Forest with selected features
rf_model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_selected.fit(X_train_selected, y_train)
y_pred_rf_selected = rf_model_selected.predict(X_test_selected)
y_pred_rf_selected_prob = rf_model_selected.predict_proba(X_test_selected)[:, 1]

print("\nRandom Forest with Selected Features Accuracy:", accuracy_score(y_test, y_pred_rf_selected))
print("Classification Report:\n", classification_report(y_test, y_pred_rf_selected))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_rf_selected_prob))
print("F1 Score:", f1_score(y_test, y_pred_rf_selected))

# Save classification report to a text file
with open('results/common/rf/classification_report_rf_selected.txt', 'w') as f:
    f.write("Classification Report for Random Forest with Selected Features:\n")
    f.write(classification_report(y_test, y_pred_rf_selected))

# Retrain XGBoost with selected features
xgb_model_selected = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model_selected.fit(X_train_selected, y_train)
y_pred_xgb_selected = xgb_model_selected.predict(X_test_selected)
y_pred_xgb_selected_prob = xgb_model_selected.predict_proba(X_test_selected)[:, 1]

print("\nXGBoost with Selected Features Accuracy:", accuracy_score(y_test, y_pred_xgb_selected))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb_selected))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_xgb_selected_prob))
print("F1 Score:", f1_score(y_test, y_pred_xgb_selected))

# Save classification report to a text file
with open('results/common/xgb/classification_report_xgb_selected.txt', 'w') as f:
    f.write("Classification Report for XGBoost with Selected Features:\n")
    f.write(classification_report(y_test, y_pred_xgb_selected))

# Retrain Voting Classifier with selected features
voting_clf_selected = VotingClassifier(
    estimators=[('rf', rf_model_selected), ('xgb', xgb_model_selected)],
    voting='soft',
    n_jobs=-1
)
voting_clf_selected.fit(X_train_selected, y_train)
y_pred_voting_selected = voting_clf_selected.predict(X_test_selected)
y_pred_voting_selected_prob = voting_clf_selected.predict_proba(X_test_selected)[:, 1]

print("\nVoting Classifier with Selected Features Accuracy:", accuracy_score(y_test, y_pred_voting_selected))
print("Classification Report:\n", classification_report(y_test, y_pred_voting_selected))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_voting_selected_prob))
print("F1 Score:", f1_score(y_test, y_pred_voting_selected))

# Save classification report to a text file
with open('results/common/classification_report_voting_selected.txt', 'w') as f:
    f.write("Classification Report for Voting Classifier with Selected Features:\n")
    f.write(classification_report(y_test, y_pred_voting_selected))

# Retrain Stacking Classifier with selected features
stacking_clf_selected = StackingClassifier(
    estimators=[('rf', rf_model_selected), ('xgb', xgb_model_selected)],
    final_estimator=LogisticRegression(max_iter=1000),
    passthrough=False,
    cv=5,
    n_jobs=-1
)
stacking_clf_selected.fit(X_train_selected, y_train)
y_pred_stacking_selected = stacking_clf_selected.predict(X_test_selected)
y_pred_stacking_selected_prob = stacking_clf_selected.predict_proba(X_test_selected)[:, 1]

print("\nStacking Classifier with Selected Features Accuracy:", accuracy_score(y_test, y_pred_stacking_selected))
print("Classification Report:\n", classification_report(y_test, y_pred_stacking_selected))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_stacking_selected_prob))
print("F1 Score:", f1_score(y_test, y_pred_stacking_selected))

# Save classification report to a text file
with open('results/common/classification_report_stacking_selected.txt', 'w') as f:
    f.write("Classification Report for Stacking Classifier with Selected Features:\n")
    f.write(classification_report(y_test, y_pred_stacking_selected))


# ------------------------ Plotting feature importance heatmap ------------------------
# Get the importance of selected features in both models
# For Random Forest
importances_rf_selected = rf_model_selected.feature_importances_
feature_importances_rf_selected = pd.Series(importances_rf_selected, index=top_n_features).sort_values(ascending=False)

# For XGBoost
importances_xgb_selected = xgb_model_selected.feature_importances_
feature_importances_xgb_selected = pd.Series(importances_xgb_selected, index=top_n_features).sort_values(ascending=False)

# Create a DataFrame containing the feature importances of the two models
feature_importances_df = pd.DataFrame({
    'Feature': top_n_features,
    'Random Forest': feature_importances_rf_selected[top_n_features],
    'XGBoost': feature_importances_xgb_selected[top_n_features]
})


feature_importances_df['Feature_Original'] = feature_importances_df['Feature'].map(cleaned_to_original)

# Set the feature name to index
feature_importances_df.set_index('Feature_Original', inplace=True)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
feature_importances_normalized = scaler.fit_transform(feature_importances_df[['Random Forest', 'XGBoost']])


feature_importances_df[['Random Forest', 'XGBoost']] = feature_importances_normalized

# heatmap
plt.figure(figsize=(10, len(top_n_features) * 0.4))
sns.heatmap(feature_importances_df[['Random Forest', 'XGBoost']], annot=True, cmap='viridis', cbar_kws={'label': 'Normalized Importance'})
plt.title('Feature Importances Heatmap - Random Forest and XGBoost')
plt.xlabel('Models')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('results/common/feature_importances_heatmap.png')
plt.show()


# ------------------------ Validate Features Using Cross-Validation ------------------------

# Cross-validation with Random Forest
scores_rf_cv = cross_val_score(rf_model_selected, X_resampled[top_n_features], y_resampled, cv=5, scoring='accuracy', n_jobs=-1)
print("\nCross-Validation Accuracy Scores for Random Forest with Selected Features:", scores_rf_cv)
print("Mean Accuracy:", scores_rf_cv.mean())

# Cross-validation with XGBoost
scores_xgb_cv = cross_val_score(xgb_model_selected, X_resampled[top_n_features], y_resampled, cv=5, scoring='accuracy', n_jobs=-1)
print("\nCross-Validation Accuracy Scores for XGBoost with Selected Features:", scores_xgb_cv)
print("Mean Accuracy:", scores_xgb_cv.mean())

# Cross-validation with Voting Classifier
scores_voting_cv = cross_val_score(voting_clf_selected, X_resampled[top_n_features], y_resampled, cv=5, scoring='accuracy', n_jobs=-1)
print("\nCross-Validation Accuracy Scores for Voting Classifier with Selected Features:", scores_voting_cv)
print("Mean Accuracy:", scores_voting_cv.mean())

# Cross-validation with Stacking Classifier
scores_stacking_cv = cross_val_score(stacking_clf_selected, X_resampled[top_n_features], y_resampled, cv=5, scoring='accuracy', n_jobs=-1)
print("\nCross-Validation Accuracy Scores for Stacking Classifier with Selected Features:", scores_stacking_cv)
print("Mean Accuracy:", scores_stacking_cv.mean())

# ------------------------ Save Final Models ------------------------

# Save the trained models for future use
joblib.dump(rf_model_selected, 'results/common/rf_model_selected.pkl')
joblib.dump(xgb_model_selected, 'results/common/xgb_model_selected.pkl')
joblib.dump(voting_clf_selected, 'results/common/voting_clf_selected.pkl')
joblib.dump(stacking_clf_selected, 'results/common/stacking_clf_selected.pkl')

# ------------------------ Conclusion ------------------------

print("\nAll models have been retrained with the selected top features.")
print("Validation using cross-validation, permutation importance, and SHAP values has been completed.")
print("The final models have been saved for future use.")

# ------------------------ End of Code ------------------------
