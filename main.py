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