# Deploying a Random Forest Model with FastAPI and Docker

This tutorial will guide you through:
1. Training a Random Forest model on the California Housing dataset
2. Saving the model as a pickle file
3. Creating a FastAPI application to serve the model
4. Containerizing the application with Docker
5. Testing the deployed model

## Prerequisites
- Python 3.10+
- Docker installed

## Step 1: Train and Save the Random Forest Model

1. Create a file called `train_model.py`:

```python
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
```

2. Run the script:
```bash
python train_model.py
```

3. Check that the pickle (named `california_housing_rf.pkl`) has been saved in the same folder as the `train_model.py`.

## Step 2: Create FastAPI Application

1. Create a file called `main.py`:

```python
# main.py
from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

# Load the model
with open('california_housing_rf.pkl', 'rb') as f:
    model = pickle.load(f)

# Define input data model
class HousingData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Create FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "California Housing Price Prediction API"}

@app.post("/predict")
def predict(data: HousingData):
    # Convert input data to numpy array
    features = np.array([
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    return {"predicted_price": prediction[0]}
```

2. Start the FastAPI server by running:
```bash
uvicorn main:app --reload
```

3. Open your web browser and navigate to the Swagger UI:
```
http://localhost:8000/docs
```

4. To test the prediction endpoint:
   - Click on the `POST /predict` endpoint
   - Click the "Try it out" button
   - Replace the example JSON with your test data:
     ```json
     {
         "MedInc": 8.3252,
         "HouseAge": 41.0,
         "AveRooms": 6.98412698,
         "AveBedrms": 1.02380952,
         "Population": 322.0,
         "AveOccup": 2.55555556,
         "Latitude": 37.88,
         "Longitude": -122.23
     }
     ```
   - Click "Execute"
   - View the server response in the "Responses" section

5. You can modify the input values and try different predictions directly from the Swagger UI interface.

6. FastAPI also provides ReDoc documentation at:
```
http://localhost:8000/redoc
```

## Step 3: Create Requirements File

Create a file called `requirements.txt`:

```
fastapi>=0.68.0
uvicorn>=0.15.0
scikit-learn>=0.24.2
numpy>=1.21.0
pydantic>=1.8.0
```

## Step 4: Create Dockerfile

Create a file called `Dockerfile`:

```dockerfile
# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Step 5: Build and Run the Docker Container

1. Build the Docker image:
```bash
docker build -t california-housing-api .
```

2. Run the container:
```bash
docker run -d -p 8000:8000 --name housing-api california-housing-api
```

## Step 6: Test the API

You can test the API using curl or Python:

### Using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.98412698,
    "AveBedrms": 1.02380952,
    "Population": 322.0,
    "AveOccup": 2.55555556,
    "Latitude": 37.88,
    "Longitude": -122.23
}'
```

### Using Python:
```python
import requests

data = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.98412698,
    "AveBedrms": 1.02380952,
    "Population": 322.0,
    "AveOccup": 2.55555556,
    "Latitude": 37.88,
    "Longitude": -122.23
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

### Expected Output
You should receive a JSON response with the predicted housing price, like:
```json
{"predicted_price":4.253}
```

## Step 7: Clean Up
To stop and remove the container:
```bash
docker stop housing-api
docker rm housing-api
```
