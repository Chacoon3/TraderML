import os
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from ML.FinNewsSentimentClassifier import FinNewsSentimentClassifier

app = FastAPI()
mlDevice= os.environ.get('ML_DEVICE', 0)
finNewsClassifier = FinNewsSentimentClassifier(mlDevice)


@app.post("/sentiment/fin-text")
async def predict_finnews_sentiment(request: Request):
    data = await request.json()
    if type(data) != list:
        return PlainTextResponse(status_code=400, content="Body should be a list of texts.")
    rawPreds = finNewsClassifier.predict(data)
    resp = [
        {
            "negative": pred[0]['score'],
            "neutral": pred[1]['score'],
            "positive": pred[1]['score']
        } for pred in rawPreds
    ]
    return JSONResponse(resp, 200)