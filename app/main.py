pip install fastapi
pip install starlette
pip install pandas
pip install joblib


# Now you can import the modules since they should be installed
from fastapi import FastAPI
from starlette.responses import JSONResponse
import pandas as pd
from joblib import load


app = FastAPI()

# Load your RandomForest model here
rfr_model = load(r'C:\Users\User\ahmedkhursheed\Deployment\rfrModelApi\models\RandomForest_model.joblib')


@app.get("/")
def read_root():
    return {"Hello": "World"}

# Remove Sales_Revenue parameter from format_features function
def format_features(
    Sales_Day: int,
    event_name: float,
    event_type: float,
    Daily_volume_of_sales: int,
    Daily_selling_price: float,
):
    return {
        'Sales_Day': Sales_Day,
        'event_name': event_name,
        'event_type': event_type,
        'Daily_volume_of_sales': Daily_volume_of_sales,
        'Daily_selling_price': Daily_selling_price,
    }

# Modify the predict function to include Sales_Revenue
@app.get("/rfr/sales/prediction")
def predict(
    Sales_Day: int,
    event_name: float,
    event_type: float,
    Daily_volume_of_sales: int,
    Daily_selling_price: float,
    Sales_Revenue: float,  # Include Sales_Revenue here
):
    features = format_features(
        Sales_Day,
        event_name,
        event_type,
        Daily_volume_of_sales,
        Daily_selling_price,
    )
    # Add Sales_Revenue to the features dictionary
    features['Sales_Revenue'] = Sales_Revenue

    obs = pd.DataFrame([features])

    # Make predictions using the loaded model
    pred = rfr_model.predict(obs)  # Use the loaded model variable name 'rfr_model'
    
    # Format predictions as a dictionary or list that can be JSON-serialized
    predictions = {"predictions": pred.tolist()}  # Example format

    return JSONResponse(content=predictions)
