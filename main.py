import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    r2_score, 
    mean_absolute_error, 
    mean_squared_error,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load only the training data
data = pd.read_csv('train.csv', header=0, index_col="row_id")

# Separate features and target
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # Last column contains bacteria names

# Convert feature columns to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Get the indices of rows without NaN values
valid_indices = ~X.isna().any(axis=1)

# Filter data using the valid indices
X_clean = X[valid_indices]
y_clean = y[valid_indices]

# Encode the target labels to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_clean)

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_encoded, test_size=0.2, random_state=42
)

# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on test data
y_pred = rf_classifier.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Generate classification report
class_report = classification_report(
    y_test, 
    y_pred, 
    target_names=label_encoder.classes_,
    zero_division=0
)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print metrics
print("\nModel Performance Metrics:")
print("-" * 30)
print(f"Accuracy: {accuracy:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

print("\nClassification Report:")
print("-" * 30)
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_clean.columns,
    'importance': rf_classifier.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importance (top 20 features)
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Top 20 Most Important Features')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Save metrics to a file
with open('model_metrics.txt', 'w') as f:
    f.write("Model Performance Metrics:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n")
    f.write(f"Mean Absolute Error: {mae:.4f}\n")
    f.write(f"Mean Squared Error: {mse:.4f}\n")
    f.write(f"Root Mean Squared Error: {rmse:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write("-" * 30 + "\n")
    f.write(class_report)

# Save the trained model to a file
joblib.dump(rf_classifier, 'random_forest_model.pkl')
print("Model saved as 'random_forest_model.pkl'")

print("\nModel training and evaluation completed.")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of test samples: {len(X_test)}")
print("Unique bacteria classes:", label_encoder.classes_)
print("\nResults have been saved to:")
print("- model_metrics.txt")
print("- confusion_matrix.png")
print("- feature_importance.png")

# Save the label encoder to a file
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Label encoder saved as 'label_encoder.pkl'")
