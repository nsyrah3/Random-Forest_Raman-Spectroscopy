import pandas as pd
import joblib

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Load the test data
test_data = pd.read_csv('test_2.csv', header=0, index_col="row_id")

# Convert feature columns to numeric
X_test = test_data.apply(pd.to_numeric, errors='coerce')

# Make predictions on test data
y_pred = model.predict(X_test)

# Convert predictions back to bacteria names
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Create a DataFrame with predictions
predictions_df = pd.DataFrame({
    'row_id': test_data.index,
    'predicted_bacteria': y_pred_labels
})

# Save predictions to CSV
predictions_df.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'") 