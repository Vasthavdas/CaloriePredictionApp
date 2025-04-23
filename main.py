from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Load model
model = joblib.load("calorie_predictor.joblib")

# Define input schema
class UserInput(BaseModel):
    age: int
    height: float
    weight: float
    duration: float
    bodytemperature: float

# Create FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Calorie Predictor API is running."}

@app.post("/predict")
def predict(data: UserInput):
    try:
        input_data = np.array([[data.age, data.height, data.weight, data.duration, data.bodytemperature]])
        prediction = model.predict(input_data)
        return {"predicted_calories": round(prediction[0], 2)}
    except Exception as e:
        return {"error": str(e)}
    # Trigger Azure redeploy
