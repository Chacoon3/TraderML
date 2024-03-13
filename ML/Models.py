from transformers import pipeline
import requests
from .Exceptions import ModelException

__name__ = "Models"


_InferenceEndpoint = "https://api-inference.huggingface.co/models/"


def _useInputConverter(**kwargs):
    if kwargs:
        def converter(dataList:list):
            return {
                "inputs": dataList,
                }
    else:
        def converter(dataList:list):
            return {
                "inputs": dataList,
                "parameters": kwargs,
                }
    return converter


def _validateResponse(resp):
    if type(resp) is dict:
        err = resp.get("error", None)
        if err != None:
            raise  ModelException(err)
    

def _useServerlessPredictor(modelId, apiToken, inputConverter = None):
    headers = {
        "Authorization": f"Bearer {apiToken}"
    }
    url = f"{_InferenceEndpoint}{modelId}"
    if inputConverter == None:
        def serverlessModel(dataList):
            response = requests.post(url, headers = headers, json = dataList)
            data = response.json()
            _validateResponse(data)
            return data
    else:
        def serverlessModel(dataList):
            data = inputConverter(dataList)
            response = requests.post(url, headers = headers, json = data)
            data = response.json()
            _validateResponse(data)
            return data
    return serverlessModel


class BaseHFEndpoint:
    """
    abstraction of hugging face endpoints
    """

    def __init__(self) -> None:
        pass

    def predict(self, dataList:list) -> list:
        pass


class SentimentClassifier(BaseHFEndpoint):

    def __init__(self, serverless: bool, apiToken = None, device = "cpu"):
        print(f"Loading {type(self)}...")

        modelId = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        if serverless == True:
            self.__predictor= _useServerlessPredictor(modelId, apiToken, _useInputConverter())
        else:
            model = pipeline("text-classification", modelId, device=device)
            if model is None:
                raise ModelException(f"Failed to load {type(self)}.")
            self.__predictor = lambda dataList: model(dataList, return_all_scores = True)

        print(f"{type(self)} loaded.")


    def predict(self, dataList: list)->list:
        preds = self.__predictor(dataList)
        res = [
            {"negative": pred[0]['score'], "neutral": pred[1]['score'], "positive": pred[2]['score']}
            for pred in preds
        ]
        return res


class TextSummarizer(BaseHFEndpoint):

    def __init__(self, serverless: bool, apiToken = None) -> None:
        print(f"Loading {type(self)}...")

        modelId = "facebook/bart-large-cnn"
        if serverless == True:
            self.__predictor = _useServerlessPredictor(modelId, apiToken, _useInputConverter(do_sample = False))   
        else:
            model = pipeline("summarization", modelId)
            if model is None:
                raise ModelException(f"Failed to load {type(self)}.")
            self.__predictor = lambda dataList: model(dataList, do_sample=False)

        print(f"{type(self)} loaded.")
    

    def predict(self, dataList: list)->list:
        preds = self.__predictor(dataList)
        res = [pred['summary_text'] for pred in preds]
        return res