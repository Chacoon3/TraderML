from transformers import pipeline
from tokenizers.pre_tokenizers import Whitespace
import requests
from .Exceptions import ModelException
from openai import OpenAI
import regex as re
from .Config import config


_InferenceEndpoint = "https://api-inference.huggingface.co/models/"
whiteSpaceTokenizer = Whitespace()
openaiClient = OpenAI(api_key = config.openaiToken)

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

    def _summarizeLengthy(dataList:list[str], maxLen:int = 512):
        pretokenized = [whiteSpaceTokenizer.pre_tokenize(data) for data in dataList]
        lengthyIndice = [i for i, p in enumerate(pretokenized) if len(p) > maxLen]
        if len(lengthyIndice) > 0:
            for index in lengthyIndice:
                raw = dataList[index]
                openaiClient.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": "You are a financial analyst."},
                        {"role": "user", "content": f"Summarize the following news. Limit your response within 500 words. \n{raw}"},
                    ]
                )
        return dataList
    

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


    def predict(self, dataList: list)->list:
        preds = self.__predictor(dataList)
        res = [
            {"negative": pred[0]['score'], "neutral": pred[1]['score'], "positive": pred[2]['score']}
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