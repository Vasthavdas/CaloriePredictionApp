from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware (this allows frontend to access backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace * with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = joblib.load("calorie_predictor.joblib")

# Define the expected input format
class UserInput(BaseModel):
    age: int
    height: float
    weight: float
    duration: float
    bodytemperature: float

# Health check route
@app.get("/")
def root():
    return {"message": "Calorie Predictor API is running."}

# Prediction route
@app.post("/predict")
def predict(data: UserInput):
    input_data = np.array([[data.age, data.height, data.weight, data.duration, data.bodytemperature]])
    prediction = model.predict(input_data)
    return {"predicted_calories": round(prediction[0], 2)}