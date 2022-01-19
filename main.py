from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str

app = FastAPI()
classifier = pipeline("sentiment-analysis")

@app.get("/")
def root():
    return {"message": "Hello UrFU"}


classifier = pipeline("sentiment-analysis",   
                      "blanchefort/rubert-base-cased-sentiment")

classifier("Я обожаю инженерию машинного обучения!")



@app.post("/predict/")
def predict(item: Item): 
    return classifier(item.text)[0]
    

   
