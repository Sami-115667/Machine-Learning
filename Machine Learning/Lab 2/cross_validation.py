import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

# Example data
X = np.random.rand(100, 1)
y = 3*X.squeeze() + np.random.randn(100)*0.1  # y = 3x + noise

# Model
model = LinearRegression()

# 5-Fold Cross Validation
kf = KFold(n_splits=15, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("Cross-validation scores:", scores)
print("Mean R^2 score:", np.mean(scores))
