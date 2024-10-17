# Install required libraries if not already installed
# Uncomment the following lines to install necessary libraries
# !pip install xgboost imbalanced-learn matplotlib seaborn pandas scikit-learn joblib

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
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

# Create the 'result' directory if it doesn't exist
os.makedirs('result', exist_ok=True)

# Load the dataset
final_data = pd.read_csv('final_data_with_long_titles.csv')

# Display the first few rows
print("First 5 rows of the dataset:")
print(final_data.head())

# Check the shape of the dataset
print(f"Dataset shape: {final_data.shape}")

# Save the initial data summary to a text file
with open('result/data_summary.txt', 'w') as f:
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

# Encode 'race' using one-hot encoding
race_dummies = pd.get_dummies(data['race'], prefix='race')
data = pd.concat([data.drop('race', axis=1), race_dummies], axis=1)

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

with open('result/class_distribution.txt', 'w') as f:
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
plt.savefig('result/class_distribution.png')
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
with open('result/resampled_class_distribution.txt', 'w') as f:
    f.write("Class distribution after resampling:\n")
    f.write(resampled_distribution.to_string())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Save the data split info to a text file
with open('result/data_split_info.txt', 'w') as f:
    f.write(f"Training set size: {X_train.shape}\n")
    f.write(f"Testing set size: {X_test.shape}\n")

# Initialize and train the XGBoost model
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Classification report
classification_rep = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(classification_rep)

# Save classification report to a text file
with open('result/classification_report.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(classification_rep)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Save confusion matrix to a CSV file
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
conf_matrix_df.to_csv('result/confusion_matrix.csv')

# Visualize Confusion Matrix with numerical labels
plt.figure(figsize=(6, 4))
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('result/confusion_matrix.png')
plt.show()

# Compute ROC AUC
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC Score: {roc_auc:.2f}")

# Save ROC AUC score to a text file
with open('result/roc_auc_score.txt', 'w') as f:
    f.write(f"ROC AUC Score: {roc_auc:.2f}\n")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('result/roc_curve.png')
plt.show()

# Compute average precision
average_precision = average_precision_score(y_test, y_proba)
print(f'Average Precision Score: {average_precision:.2f}')

# Save average precision score to a text file
with open('result/average_precision_score.txt', 'w') as f:
    f.write(f'Average Precision Score: {average_precision:.2f}\n')

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.step(recall, precision, where='post', color='b', alpha=0.7, label='Precision-Recall')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP = {average_precision:.2f})')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('result/precision_recall_curve.png')
plt.show()

# Calculate F1 Score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")

# Save F1 score to a text file
with open('result/f1_score.txt', 'w') as f:
    f.write(f"F1 Score: {f1:.2f}\n")

# Create a DataFrame with actual and predicted probabilities
lift_data = pd.DataFrame({'y_true': y_test, 'y_proba': y_proba})
lift_data = lift_data.sort_values('y_proba', ascending=False).reset_index(drop=True)
lift_data['cum_response'] = lift_data['y_true'].cumsum()
lift_data['cum_percentage'] = lift_data.index / len(lift_data)
lift_data['lift'] = lift_data['cum_response'] / (lift_data['cum_percentage'] * lift_data['y_true'].sum())

# Plot Lift Chart
plt.figure(figsize=(8, 6))
plt.plot(lift_data['cum_percentage'], lift_data['lift'], label='Lift Curve')
plt.xlabel('Percentage of Sample')
plt.ylabel('Lift')
plt.title('Lift Chart')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('result/lift_chart.png')
plt.show()

# Plot Gain Chart
plt.figure(figsize=(8, 6))
plt.plot(lift_data['cum_percentage'], lift_data['cum_response'] / lift_data['y_true'].sum(), label='Gain Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
plt.xlabel('Percentage of Sample')
plt.ylabel('Percentage of Positive Cases')
plt.title('Gain Chart')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('result/gain_chart.png')
plt.show()

# Get feature importances
feature_importances = model.feature_importances_
features = X_filled.columns.tolist()

# Create a DataFrame for feature importances
feat_imp_df = pd.DataFrame({'Feature_Cleaned': features, 'Importance': feature_importances})
# Map cleaned feature names back to original names
feat_imp_df['Feature_Original'] = feat_imp_df['Feature_Cleaned'].map(cleaned_to_original)

# Sort features by importance
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)

# Select top 20 features
top_n = 20
feat_imp_top_n = feat_imp_df.head(top_n)

# Display top features
print(f"\nTop {top_n} Most Influential Features:")
print(feat_imp_top_n[['Feature_Original', 'Importance']])

# Save feature importances to a CSV file
feat_imp_df.to_csv('result/feature_importances.csv', index=False)

# Plot Feature Importances for top 20 features without numerical labels
plt.figure(figsize=(12, 10))  # Increased figure size for better readability
ax = sns.barplot(x='Importance', y='Feature_Original', data=feat_imp_top_n, palette='viridis')
plt.title(f'Top {top_n} Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Remove numerical labels from the bars
# No annotation loop here

# Adjust y-axis labels to display fully
plt.tight_layout()
plt.savefig('result/feature_importances_top.png')
plt.show()

# Optional: Save the model
joblib.dump(model, 'result/xgboost_model.pkl')

print("\nAll steps completed successfully. Results are saved in the 'result' directory.")
