# train_model.py
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model R2 score: {score:.2f}")

# Save model
with open('california_housing_rf.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as california_housing_rf.pkl")