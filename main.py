from fastapi import FastAPI, Request
from appResponse import badFormatResponse, unhandledErrorResponse, dataResponse
from ML.Models import SentimentClassifier, TextSummarizer, BaseHFEndpoint

"""
dev note:
1. comparing inference  performance
2. use a fallback patternn to handle raw news classification
    - if the raw length is no greater than max length, analyze directly; otherwise do summarization first.\n
        - summarization: split the articles into several paragraphs and summarize each paragraph.\n
        - Join the paragraphs into one text and repeat the process till length is no greater than max length of the classifier
"""


class AppConfig:

    def __init__(self, path="credential") -> None:
        self.__config = self.readConfig(path)

    def readConfig(self, fileName):
        config = {}
        with open(fileName, "r") as f:
            for line in f.readlines():
                kvp = line.strip().split("=")
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
    

# start up
config = AppConfig()
sentimentClassifier = SentimentClassifier(config.serverless, config.apiToken, config.device)
summarizer = TextSummarizer(config.serverless, config.apiToken)
app = FastAPI()
print(config)
modelTable = dict[str,BaseHFEndpoint](
    sentiment=sentimentClassifier,
    summary = summarizer
)


@app.middleware("http")
async def exceptionHandler(request: Request, call_next):
    try:
        response = await call_next(request)
        return  response
    except Exception as e:
        return unhandledErrorResponse(e)


@app.post("/inference/{taskType}")
async def inference(request: Request, taskType:str):
    dataArray = await request.json()
    if type(dataArray) != list:
        return badFormatResponse()
    model = modelTable[taskType]
    predArray = model.predict(dataArray)
    return dataResponse(predArray)