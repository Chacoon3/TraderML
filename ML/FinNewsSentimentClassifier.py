from transformers import pipeline

# class Prediction:
#     def __init__(self, prediction = None, error = None):
#         if prediction != None and error != None:
#             raise Exception("Either prediction or error must be none.")
#         self.prediction = prediction
#         self.error = error


class FinNewsSentimentClassifier:

    def __init__(self, device):
        print(f"Loading {type(self)}...")
        self.__model = pipeline("text-classification", "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", device=device)
        if self.__model is None:
            raise Exception(f"Failed to load {type(self)}.")
        print(f"{type(self)} loaded.")

    def predict(self, newsList: list)->list:
        preds = self.__model(newsList, return_all_scores=True)
        res = [
            {"negative": pred[0]['score'], "neutral": pred[1]['score'], "positive": pred[2]['score']}
            for pred in preds
        ]
        return res
    

class FinNewsSummarizer:

    def __init__(self) -> None:
        print(f"Loading {type(self)}...")
        self.__model = pipeline("summarization", "facebook/bart-large-cnn")
        if self.__model is None:
            raise Exception(f"Failed to load {type(self)}.")
        print(f"{type(self)} loaded.")
    
    def predict(self, newsList: list)->list:
        preds = self.__model(newsList, do_sample=False)
        res = [pred['summary_text'] for pred in preds]
        return res