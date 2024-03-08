import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from appErrors import badFormatError
from ML.FinNewsSentimentClassifier import FinNewsSentimentClassifier, FinNewsSummarizer

# start up
device= os.environ.get('APP_ML_DEVICE', "cpu")
sentimentClassifier = FinNewsSentimentClassifier(device)
summarizer = FinNewsSummarizer()
app = FastAPI()


@app.post("/sentiment")
async def sentimentAnalysis(request: Request):
    data = await request.json()
    if type(data) != list:
        return badFormatError()
    preds = sentimentClassifier.predict(data)
    return JSONResponse(preds, 200)


@app.post("/summary")
async def summary(request: Request):
    data = await request.json()
    if type(data) != list:
        return badFormatError()
    resp = summarizer.predict(data)
    return JSONResponse(resp, 200)