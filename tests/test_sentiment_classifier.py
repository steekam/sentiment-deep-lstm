from src.sentiment_classifier import SentimentClassifier

def test_model_can_predict():
    classifier = SentimentClassifier()
    classifier.predict(["This is an amazing article. Thank you for sharing"])