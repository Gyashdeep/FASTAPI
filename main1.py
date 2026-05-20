import pandas as pd 
import numpy  as np 
import os 
import pickle 
from typing import List 
from fastapi import FastAPI , HTTPException 
from pydantic import BaseModel , Field 
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier 
model_path='iris_model.pkl'
if not os.path.exists(model_path):
    iris=load_iris()
    X,y=iris.data , iris.target 
    model=RandomForestClassifier(n_estimators=100 , max_depth=4 , random_state=42)
    model.fit(X,y)
    with open(model_path , 'wb') as f:
        pickle.dump({'model':model , 'target_names':iris.target_names} , f)
app=FastAPI() 
with open(model_path , 'rb') as f:
        payload=pickle.load(f)
        ml_model=payload['model']
        target_names=payload['target_names']
class IrisInput(BaseModel):
        sepal_lenth: float =Field(example=5.1)
        sepal_width: float =Field(example=3.5)
        petal__length: float =Field(example=1.5)
        petal_width: float =Field(example=0.2)
class PredictionOutput(BaseModel):
        class_index: int 
        class_label: str 
        probabilities: List[float]
@app.get('/health')
async def health_check():
    return{'status' : 'ONLINE' , 'model_load': ml_model is not None}
@app.post('/predict' , response_model=PredictionOutput)
async def predict(features: IrisInput):
    try:
        raw_features=[[
            features.sepal_length ,  
            features.sepal_width ,
            features.petal_length , 
            features.petal_wwidth 
        ]]
        prediction=int(ml_model.predict(raw_features)[0])
        probabilities=ml_model.predict_proba(raw_features)[0].tolist()
        label=str(target_names[prediction])
        return PredictionOutput(
            class_index=prediction , 
            class_label=label , 
            probabilities=probabilities 
        )
    except Exception as e:
        raise HTTPException(status_code=500 , detail=f"Inference Engine Failure: {str(e)}")
