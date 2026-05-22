from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib

app = FastAPI(
    title="Kolkata House Price Prediction API",
    description="Predict house prices in Kolkata using Gradient Boosting Regressor.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model bundle
try:
    bundle           = joblib.load("house_price_model.pkl")
    model            = bundle["model"]
    location_medians = bundle["location_medians"]
    global_median    = bundle["global_median"]
    feature_cols     = bundle["feature_cols"]
    binary_cols      = bundle["binary_cols"]
except FileNotFoundError:
    raise RuntimeError("house_price_model.pkl not found. Run house.py first.")


class HouseInput(BaseModel):
    area:     float = Field(..., gt=0)
    location: str
    bedrooms: int   = Field(..., ge=1, le=10)

    MaintenanceStaff:    int = Field(0, ge=0, le=1)
    Gymnasium:           int = Field(0, ge=0, le=1)
    SwimmingPool:        int = Field(0, ge=0, le=1)
    LandscapedGardens:   int = Field(0, ge=0, le=1)
    JoggingTrack:        int = Field(0, ge=0, le=1)
    RainWaterHarvesting: int = Field(0, ge=0, le=1)
    IndoorGames:         int = Field(0, ge=0, le=1)
    ShoppingMall:        int = Field(0, ge=0, le=1)
    Intercom:            int = Field(0, ge=0, le=1)
    SportsFacility:      int = Field(0, ge=0, le=1)
    ATM:                 int = Field(0, ge=0, le=1)
    ClubHouse:           int = Field(0, ge=0, le=1)
    School:              int = Field(0, ge=0, le=1)
    Security24X7:        int = Field(0, ge=0, le=1)
    PowerBackup:         int = Field(0, ge=0, le=1)
    CarParking:          int = Field(0, ge=0, le=1)
    StaffQuarter:        int = Field(0, ge=0, le=1)
    Cafeteria:           int = Field(0, ge=0, le=1)
    MultipurposeRoom:    int = Field(0, ge=0, le=1)
    Hospital:            int = Field(0, ge=0, le=1)
    WashingMachine:      int = Field(0, ge=0, le=1)
    Gasconnection:       int = Field(0, ge=0, le=1)
    AC:                  int = Field(0, ge=0, le=1)
    Wifi:                int = Field(0, ge=0, le=1)
    ChildrensPlayarea:   int = Field(0, ge=0, le=1)
    LiftAvailable:       int = Field(0, ge=0, le=1)
    BED:                 int = Field(0, ge=0, le=1)
    VaastuCompliant:     int = Field(0, ge=0, le=1)
    Microwave:           int = Field(0, ge=0, le=1)
    GolfCourse:          int = Field(0, ge=0, le=1)
    TV:                  int = Field(0, ge=0, le=1)
    DiningTable:         int = Field(0, ge=0, le=1)
    Sofa:                int = Field(0, ge=0, le=1)
    Wardrobe:            int = Field(0, ge=0, le=1)
    Refrigerator:        int = Field(0, ge=0, le=1)
    Resale:              int = Field(0, ge=0, le=1)


class PredictionResult(BaseModel):
    predicted_price:       float
    predicted_price_lakhs: float
    price_per_sqft:        float
    price_range_low:       float
    price_range_high:      float
    amenity_score:         int
    location_used:         str
    note:                  str


@app.get("/")
def root():
    return {"message": "Kolkata House Price API running. Visit /docs for documentation."}

@app.get("/health")
def health():
    return {"status": "ok", "locations": int(len(location_medians))}

@app.get("/locations")
def get_locations():
    return {"locations": sorted(location_medians.index.tolist())}


@app.post("/predict", response_model=PredictionResult)
def predict(data: HouseInput):
    try:
        # Map clean field names back to dataset column names
        bin_vals = {
            'MaintenanceStaff':    data.MaintenanceStaff,
            'Gymnasium':           data.Gymnasium,
            'SwimmingPool':        data.SwimmingPool,
            'LandscapedGardens':   data.LandscapedGardens,
            'JoggingTrack':        data.JoggingTrack,
            'RainWaterHarvesting': data.RainWaterHarvesting,
            'IndoorGames':         data.IndoorGames,
            'ShoppingMall':        data.ShoppingMall,
            'Intercom':            data.Intercom,
            'SportsFacility':      data.SportsFacility,
            'ATM':                 data.ATM,
            'ClubHouse':           data.ClubHouse,
            'School':              data.School,
            '24X7Security':        data.Security24X7,
            'PowerBackup':         data.PowerBackup,
            'CarParking':          data.CarParking,
            'StaffQuarter':        data.StaffQuarter,
            'Cafeteria':           data.Cafeteria,
            'MultipurposeRoom':    data.MultipurposeRoom,
            'Hospital':            data.Hospital,
            'WashingMachine':      data.WashingMachine,
            'Gasconnection':       data.Gasconnection,
            'AC':                  data.AC,
            'Wifi':                data.Wifi,
            "Children'splayarea":  data.ChildrensPlayarea,
            'LiftAvailable':       data.LiftAvailable,
            'BED':                 data.BED,
            'VaastuCompliant':     data.VaastuCompliant,
            'Microwave':           data.Microwave,
            'GolfCourse':          data.GolfCourse,
            'TV':                  data.TV,
            'DiningTable':         data.DiningTable,
            'Sofa':                data.Sofa,
            'Wardrobe':            data.Wardrobe,
            'Refrigerator':        data.Refrigerator,
            'Resale':              data.Resale,
        }

        amenity_score    = sum(bin_vals.values())
        area_per_bedroom = data.area / max(data.bedrooms, 1)
        loc_encoded      = float(location_medians.get(data.location, global_median))

        row = {
            'Area':            data.area,
            'Location':        loc_encoded,
            'No. of Bedrooms': data.bedrooms,
            'AreaPerBedroom':  area_per_bedroom,
            'AmenityScore':    amenity_score,
            **bin_vals
        }

        X_input   = pd.DataFrame([row])[feature_cols]
        predicted = float(model.predict(X_input)[0])
        predicted = max(predicted, 0)

        loc_label = data.location if data.location in location_medians.index \
                    else f"{data.location} (unknown — used global median)"

        return PredictionResult(
            predicted_price=round(predicted, 2),
            predicted_price_lakhs=round(predicted / 1e5, 2),
            price_per_sqft=round(predicted / data.area, 2),
            price_range_low=round(predicted * 0.88 / 1e5, 2),
            price_range_high=round(predicted * 1.12 / 1e5, 2),
            amenity_score=amenity_score,
            location_used=loc_label,
            note="Predicted using Gradient Boosting Regressor trained on Kolkata housing data."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))