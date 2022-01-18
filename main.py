from fastapi import FastAPI
import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
classifier = pipeline("sentiment-analysis")

@app.get("/")
def root():
    return {"message": "Hello UrFU"}

@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]
