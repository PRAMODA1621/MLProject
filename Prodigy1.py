import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('housing.csv')
data = data.dropna()

# Add interaction features (e.g., bedrooms * area, bathrooms * stories)
data['bedrooms_area'] = data['bedrooms'] * data['area']
data['bathrooms_stories'] = data['bathrooms'] * data['stories']

# Include other relevant features
X = data[['bedrooms', 'bathrooms', 'area', 'stories', 'bedrooms_area', 'bathrooms_stories']].values
Y = data['price'].values

# Log transformation of target variable to reduce skewness
Y_log = np.log(Y)  # Apply log transformation to target variable

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_log, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Gradient Boosting Regressor model
model = GradientBoostingRegressor(random_state=42)

# Define parameter grid for GridSearchCV (tuned range for n_estimators and max_depth)
param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'subsample': [0.7, 0.8, 0.9],  # Subsample for controlling overfitting
    'min_samples_split': [2, 5, 10]  # Min samples split for more control
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
mae = mean_absolute_error(np.exp(Y_test), np.exp(y_pred_test))  # Reverse log transform for MAE calculation
rmse = np.sqrt(mean_squared_error(np.exp(Y_test), np.exp(y_pred_test)))
r2 = r2_score(np.exp(Y_test), np.exp(y_pred_test))  # Reverse log transform for R-squared

# Print results
print("\nPredictions on test data:")
for i, prediction in enumerate(np.exp(y_pred_test)):  # Reverse log transform for predictions
    print(f"House {i+1}: Predicted Price = ${prediction:,.2f}, Actual Price = ${np.exp(Y_test[i]):,.2f}")

print(f"\nMAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R-squared: {r2:.4f}")
print(f"Best Parameters from Grid Search: {grid_search.best_params_}")
