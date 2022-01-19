from fastapi import FastAPI
import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast




app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello UrFU"}

@app.post("/predict/")
def predict(item: text):
    return classifier(item.text)[0]
