import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Example data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])  # y = x^2 pattern

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train linear regression on polynomial features
model = LinearRegression()
model.fit(X_poly, y)

# Predict
y_pred = model.predict(X_poly)

# Plot
plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
plt.show()
plt.show()
