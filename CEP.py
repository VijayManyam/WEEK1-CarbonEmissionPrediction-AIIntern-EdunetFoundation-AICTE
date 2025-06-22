# Carbon Emission Prediction (Clean & Beginner Friendly Version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "E:/PROJECTS/IBM-CARBON_EMMISION_PREDICTION/data_cleaned.csv"
data = pd.read_csv(file_path)

# Show dataset info
print("\n📊 Dataset Preview:")
print(data.head())
print(f"\n🔢 Original Shape: {data.shape}")

# Drop missing values
data = data.dropna()
print(f"✅ Shape After Dropping Missing Values: {data.shape}")

# Drop non-numeric or unnecessary columns
data = data.drop(columns=['country'])

# Set feature variables and target variable
target_column = 'co2_ttl'  # We are predicting total CO2 emissions
X = data.drop(columns=[target_column])  # All other columns as input
y = data[target_column]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale the input features (important for linear models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📉 Mean Squared Error: {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Save model and scaler
joblib.dump(model, "carbon_emission_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n💾 Model and Scaler saved successfully.")

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='darkblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'red', linestyle='--')
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Actual vs Predicted CO2 Emissions")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color='orange')
plt.title("Distribution of Prediction Errors (Residuals)")
plt.xlabel("Error")
plt.grid(True)
plt.tight_layout()
plt.show()
