from transformers import pipeline

class FinNewsSentimentClassifier:
    def __init__(self, device = 0):
        self.__model = pipeline("text-classification", "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", device=device)

    def predict(self, news)->list:
        return self.__model(news, return_all_scores=True)