import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Step 1: Load the data
data = pd.read_csv("Automobile price data _Raw_.csv")

# Step 2: Handle missing values and drop unnecessary columns
data.replace("?", pd.NA, inplace=True)
data.dropna(subset=['make', 'body-style', 'wheel-base', 'engine-size', 'horsepower', 'peak-rpm', 'highway-mpg', 'price'], inplace=True)

# Step 3: Select only the relevant columns
data = data[['make', 'body-style', 'wheel-base', 'engine-size', 'horsepower', 'peak-rpm', 'highway-mpg', 'price']]

# Step 4: Separate features and target variable
X = data.drop(columns='price')
y = data['price'].astype(float)

# Step 5: Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Step 6: Define preprocessing pipelines
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 7: Create Linear Regression pipeline
linear_regression_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Step 8: Create Gradient Boosting Regressor pipeline
gradient_boosting_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

# Step 9: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 10: Train Linear Regression model
linear_regression_model.fit(X_train, y_train)

# Step 11: Train Gradient Boosting Regressor model
gradient_boosting_model.fit(X_train, y_train)

# Step 12: Make predictions with both models
y_pred_lr = linear_regression_model.predict(X_test)
y_pred_gb = gradient_boosting_model.predict(X_test)

# Step 13: Calculate and print errors for Linear Regression model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
rae_lr = np.sum(np.abs(y_test - y_pred_lr)) / np.sum(np.abs(y_test - np.mean(y_test)))
rse_lr = np.sum((y_test - y_pred_lr) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print(f"Linear Regression - Mean Absolute Error (MAE): {mae_lr:.4f}")
print(f"Linear Regression - Root Mean Squared Error (RMSE): {rmse_lr:.4f}")
print(f"Linear Regression - Relative Absolute Error (RAE): {rae_lr:.4f}")
print(f"Linear Regression - Relative Squared Error (RSE): {rse_lr:.4f}")
print(f"Linear Regression - Coefficient of Determination (R^2): {r2_lr:.4f}")

# Step 14: Calculate and print errors for Gradient Boosting Regressor model
mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = mean_squared_error(y_test, y_pred_gb, squared=False)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
rae_gb = np.sum(np.abs(y_test - y_pred_gb)) / np.sum(np.abs(y_test - np.mean(y_test)))
rse_gb = np.sum((y_test - y_pred_gb) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print(f"\nGradient Boosting - Mean Absolute Error (MAE): {mae_gb:.4f}")
print(f"Gradient Boosting - Root Mean Squared Error (RMSE): {rmse_gb:.4f}")
print(f"Gradient Boosting - Relative Absolute Error (RAE): {rae_gb:.4f}")
print(f"Gradient Boosting - Relative Squared Error (RSE): {rse_gb:.4f}")
print(f"Gradient Boosting - Coefficient of Determination (R^2): {r2_gb:.4f}")

# Step 15: Combine predictions
combined_predictions = (0.45 * y_pred_lr + 0.55 * y_pred_gb)

# Step 16: Calculate and print errors for the Combined Model
mae_combined = mean_absolute_error(y_test, combined_predictions)
rmse_combined = mean_squared_error(y_test, combined_predictions, squared=False)
mse_combined = mean_squared_error(y_test, combined_predictions)
r2_combined = r2_score(y_test, combined_predictions)
rae_combined = np.sum(np.abs(y_test - combined_predictions)) / np.sum(np.abs(y_test - np.mean(y_test)))
rse_combined = np.sum((y_test - combined_predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print(f"\nCombined Model - Mean Absolute Error (MAE): {mae_combined:.4f}")
print(f"Combined Model - Root Mean Squared Error (RMSE): {rmse_combined:.4f}")
print(f"Combined Model - Relative Absolute Error (RAE): {rae_combined:.4f}")
print(f"Combined Model - Relative Squared Error (RSE): {rse_combined:.4f}")
print(f"Combined Model - Coefficient of Determination (R^2): {r2_combined:.4f}")

# Step 17: Create a DataFrame to compare actual vs. predicted prices for both models
comparison_df = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price (Linear Regression)': y_pred_lr,
    'Predicted Price (Gradient Boosting)': y_pred_gb,
    'Combined Predicted Price': combined_predictions
})

print("\nComparison of Actual and Predicted Prices:\n", comparison_df.head())

# Step 18: Plotting

plt.figure(figsize=(18, 6))

# Scatter plot for Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.5, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price (Linear Regression)')
plt.title('Actual vs Predicted Prices (Linear Regression)')
plt.legend()

# Scatter plot for Gradient Boosting
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_gb, alpha=0.5, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price (Gradient Boosting)')
plt.title('Actual vs Predicted Prices (Gradient Boosting)')
plt.legend()

# Scatter plot for Combined Model
plt.subplot(1, 3, 3)
plt.scatter(y_test, combined_predictions, alpha=0.5, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Price')
plt.ylabel('Combined Predicted Price')
plt.title('Combined Model')
plt.legend()

plt.tight_layout()
plt.show()

# Plot prediction errors
plt.figure(figsize=(18, 6))

# Error for Linear Regression
plt.subplot(1, 3, 1)
errors_lr = y_test - y_pred_lr
sns.histplot(errors_lr, kde=True)
plt.xlabel('Prediction Error (Linear Regression)')
plt.title('Distribution of Prediction Errors (Linear Regression)')

# Error for Gradient Boosting
plt.subplot(1, 3, 2)
errors_gb = y_test - y_pred_gb
sns.histplot(errors_gb, kde=True)
plt.xlabel('Prediction Error (Gradient Boosting)')
plt.title('Distribution of Prediction Errors (Gradient Boosting)')

# Error for Combined Model
plt.subplot(1, 3, 3)
errors_combined = y_test - combined_predictions
sns.histplot(errors_combined, kde=True)
plt.xlabel('Prediction Error (Combined Model)')
plt.title('Distribution of Prediction Errors (Combined Model)')

plt.tight_layout()
plt.show()

