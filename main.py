from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from appErrors import badFormatError, unhandledError
from ML.Models import SentimentClassifier, TextSummarizer


# start up
class AppConfig:

    def __init__(self, path="credential") -> None:
        self.__config = self.readConfig(path)

    def readConfig(self, fileName):
        config = {}
        with open(fileName, "r") as f:
            for line in f.readlines():
                kvp = line.split("=")
                config[kvp[0]] = kvp[1]
        return config
    
    @property
    def device(self):
        return self.__config.get("device", "cpu")
    
    @property
    def serverless(self):
        return self.__config.get("serverless", 1)
    
    @property
    def apiToken(self):
        return self.__config.get("apiToken", None)

    def __str__(self) -> str:
        specified = "specified" if self.apiToken != None else "unspecified"
        return f"App Config:\ndevice: {self.device}\nserverless:{self.serverless}\ntoken:{specified}"
    

config = AppConfig()
sentimentClassifier = SentimentClassifier(config.serverless, config.apiToken, config.device)
summarizer = TextSummarizer(config.serverless, config.apiToken)
app = FastAPI()
print(config)


@app.middleware("http")
async def exceptionHandler(request: Request, call_next):
    try:
        response = await call_next(request)
        return  response
    except Exception as e:
        return unhandledError(e)


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