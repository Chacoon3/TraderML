import os
from fastapi import FastAPI, Request, HTTPException
from ML.FinNewsSentimentClassifier import FinNewsSentimentClassifier
import json

app = FastAPI()
mlDevice= os.environ.get('ML_DEVICE', 0)
finNewsClassifier = FinNewsSentimentClassifier(mlDevice)


@app.post("/sentiment/fin-text")
def predict_finnews_sentiment(request: Request):
    newsList = json.loads(request.body())
    if not newsList is list:
        return HTTPException(status_code=400, detail="Body should be a list of texts.")
    rawPreds = finNewsClassifier.predict(newsList)
    resp = [
        {
            "negative": pred[0]['score'],
            "neutral": pred[1]['score'],
            "positive": pred[1]['score']
        } for pred in rawPreds
    ]
    return resp
    

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}