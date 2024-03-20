from fastapi import FastAPI, Request
from App.Response import badFormatResponse, unhandledErrorResponse, dataResponse
from App.Models import SentimentClassifier, TextSummarizer, BaseHFEndpoint
from App.Config import config

"""
dev note:
1. comparing inference  performance
2. use a fallback patternn to handle raw news classification
    - if the raw length is no greater than max length, analyze directly; otherwise do summarization first.\n
        - summarization: split the articles into several paragraphs and summarize each paragraph.\n
        - Join the paragraphs into one text and repeat the process till length is no greater than max length of the classifier
3. using GPT to summarize the raw texts
"""

    

# start up
sentimentClassifier = SentimentClassifier(config.serverless, config.hfToken, config.device)
summarizer = TextSummarizer(config.serverless, config.hfToken, config.device)
app = FastAPI()
print(config)
modelTable = dict[str,BaseHFEndpoint](
    sentiment=sentimentClassifier,
    summary = summarizer
)


@app.middleware("http")
async def errorMiddleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return  response
    except Exception as e:
        return unhandledErrorResponse(e)


@app.post("/inference/{taskType}")
async def inference(request: Request, taskType:str):
    # abstraction of inference performed on array inputs
    dataArray = await request.json()
    if type(dataArray) != list:
        return badFormatResponse()
    model = modelTable[taskType]
    predArray = model.predict(dataArray)
    return dataResponse(predArray)