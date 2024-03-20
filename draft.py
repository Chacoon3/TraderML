from App.Models import SentimentClassifier
from App.Config import config

foo = SentimentClassifier(True, config.hfToken, config.device)

preds = foo.predict(["I am happy", "I am sad"])

print(preds)