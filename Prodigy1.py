import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('data.csv')
data = data.dropna()

# Preprocess categorical features (assume they are categorical)
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(data[['Engine Fuel Type', 'Engine HP']])  # Assuming 'Engine Fuel Type' and 'Engine HP' are categorical
X = np.hstack((X_encoded, data[['Engine Cylinders', 'Year']].values))  # Include numerical features such as Engine Cylinders and Year
Y = data['MSRP'].values

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.001, 0.01, 0.1]
}

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, Y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train_scaled, Y_train)

# Make predictions on the test set
y_pred_test = best_model.predict(X_test_scaled)

# Calculate evaluation metrics
mae = mean_absolute_error(Y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred_test))
r2 = r2_score(Y_test, y_pred_test)

# Print results
print("\nPredictions on test data:")
for i, prediction in enumerate(y_pred_test):  # Show first 10 predictions for brevity
    print(f"House {i+1}: Predicted Price = ${prediction:,.2f}, Actual Price = ${Y_test[i]:,.2f}")

print(f"\nMAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R-squared: {r2:.4f}")
print(f"Best Parameters from Grid Search: {grid_search.best_params_}")
