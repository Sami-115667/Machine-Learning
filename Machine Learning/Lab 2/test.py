import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Helper Class: Linear Regression
# ----------------------------
class LinearRegressionGD:
    def __init__(self, alpha=0.01, iterations=1000):
        self.alpha = alpha
        self.iterations = iterations
        self.theta = None
        self.train_errors = []
        self.val_errors = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        m, n = X_train.shape
        self.theta = np.zeros(n)
        
        for i in range(self.iterations):
            y_pred = X_train.dot(self.theta)
            error = y_pred - y_train
            grad = (1/m) * X_train.T.dot(error)
            self.theta -= self.alpha * grad

            train_error = (1/(2*m)) * np.sum(error**2)
            self.train_errors.append(train_error)

            if X_val is not None and y_val is not None:
                val_error = (1/(2*X_val.shape[0])) * np.sum((X_val.dot(self.theta) - y_val)**2)
                self.val_errors.append(val_error)

    def predict(self, X):
        return X.dot(self.theta)
    
    def plot_errors(self):
        plt.plot(self.train_errors, label="Training Error")
        if self.val_errors:
            plt.plot(self.val_errors, label="Validation Error")
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.legend()
        plt.title('Training and Validation Error Curves')
        plt.show()

# ----------------------------
# Helper Function: Create Polynomial Features
# ----------------------------
def create_polynomial_features(X, degree):
    X_poly = np.ones((X.shape[0], 1))
    for d in range(1, degree+1):
        X_poly = np.hstack((X_poly, X**d))
    return X_poly

# ----------------------------
# (A) Linear Regression with Multiple Variables
# ----------------------------

# Load Data
data = pd.read_csv('data_02a.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Plot Features vs Target
for i in range(X.shape[1]):
    plt.scatter(X[:, i], y)
    plt.xlabel(f'Feature {i+1}')
    plt.ylabel('Target')
    plt.title(f'Feature {i+1} vs Target')
    plt.show()

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Add Bias Term
X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_val_bias = np.c_[np.ones(X_val.shape[0]), X_val]

# Linear Regression without Scaling
model = LinearRegressionGD(alpha=0.01, iterations=500)
model.fit(X_train_bias, y_train, X_val_bias, y_val)
model.plot_errors()

print("\n--- Linear Regression Without Normalization ---")
print(f"Best Training Error: {min(model.train_errors):.4f}")
print(f"Best Validation Error: {min(model.val_errors):.4f}")
print(f"Learned Parameters: {model.theta}")

# Linear Regression with Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_scaled_bias = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_val_scaled_bias = np.c_[np.ones(X_val_scaled.shape[0]), X_val_scaled]

model_scaled = LinearRegressionGD(alpha=0.01, iterations=500)
model_scaled.fit(X_train_scaled_bias, y_train, X_val_scaled_bias, y_val)
model_scaled.plot_errors()

print("\n--- Linear Regression With Normalization ---")
print(f"Best Training Error: {min(model_scaled.train_errors):.4f}")
print(f"Best Validation Error: {min(model_scaled.val_errors):.4f}")
print(f"Learned Parameters: {model_scaled.theta}")

# ----------------------------
# (B) 5-Fold Cross Validation
# ----------------------------

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_train_errors = []
fold_val_errors = []
fold_thetas = []

for train_idx, val_idx in kf.split(X):
    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
    
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_val_cv_scaled = scaler_cv.transform(X_val_cv)
    
    X_train_cv_scaled_bias = np.c_[np.ones(X_train_cv_scaled.shape[0]), X_train_cv_scaled]
    X_val_cv_scaled_bias = np.c_[np.ones(X_val_cv_scaled.shape[0]), X_val_cv_scaled]
    
    model_cv = LinearRegressionGD(alpha=0.01, iterations=500)
    model_cv.fit(X_train_cv_scaled_bias, y_train_cv, X_val_cv_scaled_bias, y_val_cv)
    
    fold_train_errors.append(min(model_cv.train_errors))
    fold_val_errors.append(min(model_cv.val_errors))
    fold_thetas.append(model_cv.theta)

print("\n--- 5-Fold Cross Validation Results ---")
print(f"Average Training Error: {np.mean(fold_train_errors):.4f}")
print(f"Average Validation Error: {np.mean(fold_val_errors):.4f}")

# ----------------------------
# (C) Polynomial Regression (Single Variable)
# ----------------------------

# Load Data
data_poly = pd.read_csv('data_02b.csv')
X_poly = data_poly.iloc[:, 0].values.reshape(-1, 1)
y_poly = data_poly.iloc[:, 1].values

# Plot Feature vs Target
plt.scatter(X_poly, y_poly)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Feature vs Target (Polynomial Regression)')
plt.show()

degrees = [1, 2, 3]
val_errors_poly = []

# Train and Plot for each degree
for d in degrees:
    X_poly_d = create_polynomial_features(X_poly, d)
    
    X_train_poly, X_val_poly, y_train_poly, y_val_poly = train_test_split(X_poly_d, y_poly, test_size=0.2, random_state=42)
    
    model_d = LinearRegressionGD(alpha=0.01, iterations=500)
    model_d.fit(X_train_poly, y_train_poly, X_val_poly, y_val_poly)
    
    val_errors_poly.append(min(model_d.val_errors))
    
    x_range = np.linspace(X_poly.min(), X_poly.max(), 100).reshape(-1, 1)
    x_range_poly = create_polynomial_features(x_range, d)
    y_pred_curve = model_d.predict(x_range_poly)
    
    plt.plot(x_range, y_pred_curve, label=f'Degree {d}')

# Final Plot: All fitted curves
plt.scatter(X_poly, y_poly, color='black', label='Data')
plt.legend()
plt.title('Polynomial Regression Fits')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()

# Bar Plot: Validation Errors
plt.bar([str(d) for d in degrees], val_errors_poly)
plt.xlabel('Degree')
plt.ylabel('Validation Error')
plt.title('Validation Error vs Polynomial Degree')
plt.show()

# Report Best Degree
best_degree = degrees[np.argmin(val_errors_poly)]
print("\n--- Polynomial Regression Best Degree ---")
print(f"Best Degree: {best_degree}")
