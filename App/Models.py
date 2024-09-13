from transformers import pipeline
from tokenizers.pre_tokenizers import Whitespace
import requests
from .Exceptions import ModelException
import regex as re
from sklearn.feature_extraction.text import TfidfTransformer


_InferenceEndpoint = "https://api-inference.huggingface.co/models/"
whiteSpaceTokenizer = Whitespace()

def _useTextualInputProcessor(**kwargs):
    """
    convert raw input into the format required by huggingface serverless inference endpoint
    """

    def _removeHtmlElements(text):
        # remove html elements
        processed = re.sub(re.compile(r"<.*?>"), "", text)
        # remove special chars
        processed = re.sub(r"&amp;", "&", processed)
        processed = re.sub(r"\s+|&#[0-9]+;|&nbsp;", " ", processed)
        return processed
    

    if not kwargs:
        def converter(dataList:list):
            dataList = [_removeHtmlElements(data) for data in dataList]
            return {
                "inputs": dataList,
                }
    else:
        def converter(dataList:list):
            dataList = [_removeHtmlElements(data) for data in dataList]
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
    """
    wrapping of huggingface serverless inference endpoint
    """

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
            self.__predictor= _useServerlessPredictor(modelId, apiToken, _useTextualInputProcessor(truncation = True))
        else:
            model = pipeline("text-classification", modelId, device=device, truncation=True)
            if model is None:
                raise ModelException(f"Failed to load {type(self)}.")
            self.__predictor = lambda dataList: model(dataList, return_all_scores = True)

        print(f"{type(self)} loaded.")

    
    def __convertToDict(self, pred:list[dict]):
        res = dict[str, int](negative = 0, neutral = 0, positive = 0)
        for p in pred:
            if p['label'] == "negative":
                res["negative"] = p['score']
            elif p['label'] == "neutral":
                res["neutral"] = p['score']
            else:
                res["positive"] = p['score']
        return res
        


    def predict(self, dataList: list)->list:
        preds = self.__predictor(dataList)
        res = [
            self.__convertToDict(pred)
            for pred in preds
        ]
        return res


class TextSummarizer(BaseHFEndpoint):

    def __init__(self, serverless: bool, apiToken = None, device="cpu") -> None:
        print(f"Loading {type(self)}...")

        modelId = "facebook/bart-large-cnn"
        if serverless == True:
            self.__predictor = _useServerlessPredictor(modelId, apiToken, _useTextualInputProcessor(do_sample = False, truncation = True))   
        else:
            model = pipeline("summarization", modelId, device=device, truncation=True)
            if model is None:
                raise ModelException(f"Failed to load {type(self)}.")
            self.__predictor = lambda dataList: model(dataList, do_sample=False)

        print(f"{type(self)} loaded.")
    

    def predict(self, dataList: list)->list:
        preds = self.__predictor(dataList)
        res = [pred['summary_text'] for pred in preds]
        return res
    

class TfIdfPredictor(BaseHFEndpoint):

    def __init__(self) -> None:
        super().__init__()

    
    def predict(self, dataList: list[list])->list:
        transformer = TfidfTransformer()
        res = transformer.fit_transform(dataList)
        return res