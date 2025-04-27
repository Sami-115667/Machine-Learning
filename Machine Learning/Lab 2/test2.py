import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold

# ----------------------------
# (A) Linear Regression with Multiple Variables
# ----------------------------

# Load data
data = pd.read_csv('data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Plot Features vs Target
for i in range(X.shape[1]):
    plt.scatter(X[:, i], y)
    plt.xlabel(f'Feature {i+1}')
    plt.ylabel('Target')
    plt.title(f'Feature {i+1} vs Target')
    plt.show()

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression without Scaling
model = LinearRegression()
model.fit(X_train, y_train)

train_error = np.mean((model.predict(X_train) - y_train) ** 2)
val_error = np.mean((model.predict(X_val) - y_val) ** 2)

print("\n--- Linear Regression Without Normalization ---")
print(f"Training Error: {train_error:.4f}")
print(f"Validation Error: {val_error:.4f}")
print(f"Learned Parameters: {np.append(model.intercept_, model.coef_)}")

# Linear Regression with Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)

train_error_scaled = np.mean((model_scaled.predict(X_train_scaled) - y_train) ** 2)
val_error_scaled = np.mean((model_scaled.predict(X_val_scaled) - y_val) ** 2)

print("\n--- Linear Regression With Normalization ---")
print(f"Training Error: {train_error_scaled:.4f}")
print(f"Validation Error: {val_error_scaled:.4f}")
print(f"Learned Parameters: {np.append(model_scaled.intercept_, model_scaled.coef_)}")

# ----------------------------
# (B) 5-Fold Cross Validation
# ----------------------------

kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_errors = []
val_errors = []

for train_idx, val_idx in kf.split(X):
    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
    
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_val_cv_scaled = scaler_cv.transform(X_val_cv)
    
    model_cv = LinearRegression()
    model_cv.fit(X_train_cv_scaled, y_train_cv)
    
    train_errors.append(np.mean((model_cv.predict(X_train_cv_scaled) - y_train_cv) ** 2))
    val_errors.append(np.mean((model_cv.predict(X_val_cv_scaled) - y_val_cv) ** 2))

print("\n--- 5-Fold Cross Validation Results ---")
print(f"Average Training Error: {np.mean(train_errors):.4f}")
print(f"Average Validation Error: {np.mean(val_errors):.4f}")

# ----------------------------
# (C) Polynomial Regression (Single Variable)
# ----------------------------

# Load polynomial data
data_poly = pd.read_csv('data_02b.csv')
X_poly = data_poly.iloc[:, 0].values.reshape(-1, 1)
y_poly = data_poly.iloc[:, 1].values

# Plot feature
plt.scatter(X_poly, y_poly)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Feature vs Target (Polynomial)')
plt.show()

degrees = [1, 2, 3]
val_errors_poly = []
models_poly = []

# Train for each degree
for degree in degrees:
    poly = PolynomialFeatures(degree)
    X_poly_transformed = poly.fit_transform(X_poly)
    
    X_train_poly, X_val_poly, y_train_poly, y_val_poly = train_test_split(X_poly_transformed, y_poly, test_size=0.2, random_state=42)
    
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train_poly)
    
    val_pred = model_poly.predict(X_val_poly)
    val_error = np.mean((val_pred - y_val_poly) ** 2)
    
    val_errors_poly.append(val_error)
    models_poly.append((model_poly, poly))

# Plot all fitted curves
X_range = np.linspace(X_poly.min(), X_poly.max(), 100).reshape(-1, 1)
plt.scatter(X_poly, y_poly, color='black', label='Data')

for model_poly, poly in models_poly:
    X_range_poly = poly.transform(X_range)
    y_range_pred = model_poly.predict(X_range_poly)
    plt.plot(X_range, y_range_pred, label=f'Degree {poly.degree}')

plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Polynomial Fits')
plt.legend()
plt.show()

# Bar plot for validation errors
plt.bar([str(d) for d in degrees], val_errors_poly)
plt.xlabel('Degree')
plt.ylabel('Validation Error')
plt.title('Validation Error vs Polynomial Degree')
plt.show()

# Best degree
best_degree_idx = np.argmin(val_errors_poly)
print("\n--- Polynomial Regression Best Degree ---")
print(f"Best Degree: {degrees[best_degree_idx]}")
