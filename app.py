from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Annotated
import pandas as pd
import pickle

api = FastAPI()

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Pydantic input schema
class Car(BaseModel):
    Brand: Annotated[str, Field(..., description="Give the Brand of the car")]
    Year: Annotated[int, Field(..., description="Model year of the car")]
    Age: Annotated[int, Field(..., description="How much age of the car")]
    kmDriven: Annotated[int, Field(..., description="Enter the km of car driven")]
    Transmission: Annotated[str, Field(..., description="Enter the car transmission: Manual or Automatic")]
    Owner: Annotated[str, Field(..., description="Give type of owner")]
    FuelType: Annotated[str, Field(..., description="Enter the type of fuel")]

# Prediction endpoint
@api.post('/predict')
def predict_price(data: Car):
    input_df = pd.DataFrame([{
        'Brand': data.Brand,
        'Year': data.Year,
        'Age': data.Age,
        'kmDriven': data.kmDriven,
        'Transmission': data.Transmission,
        'Owner': data.Owner,
        'FuelType': data.FuelType
    }])

    prediction = model.predict(input_df)[0]
    # prediction = max(0, round(prediction))  # prevent negative price

    return JSONResponse(status_code=200, content={'predicted_price': round(abs(prediction),0)})
