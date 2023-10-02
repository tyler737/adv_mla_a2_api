from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

sgd_pipe = load('../models/sgd_pipeline.joblib')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'SGDRegressor is all ready to go!'

def format_features(
    item_id: int,
    dept_id: int,
    store_id: int,
    cat_id: int,
    state_id: int,
    event_name: int,
    event_type: int,
    wday: int,
    month: int,
    year: int,
    week: int,
    ):
    return {
        'item_id': [item_id],
        'dept_id': [dept_id],
        'store_id': [store_id],
        'cat_id': [cat_id],
        'state_id': [state_id],
        'event_name': [event_name],
        'event_type': [event_type],
        'wday': [wday],
        'month': [month],
        'year': [year],
        'week': [week]
    }

@app.get("/sales/stores/items/")
def predict(
    item_id: int,
    dept_id: int,
    store_id: int,
    cat_id: int,
    state_id: int,
    event_name: int,
    event_type: int,
    wday: int,
    month: int,
    year: int,
    week: int,
):
    features = format_features(
        item_id,
        dept_id,
        store_id,
        cat_id,
        state_id,
        event_name,
        event_type,
        wday,
        month,
        year,
        week
        )
    obs = pd.DataFrame(features)
    pred = sgd_pipe.predict(obs)
    return JSONResponse(pred.tolist())
