from typing import List
from fastapi import FastAPI
from app.schemas import Comment, ClassifiedComment
from app.sentiment_classifier import SentimentClassifier

app = FastAPI()

classifier = SentimentClassifier()


@app.post("/classify", response_model=List[ClassifiedComment])
async def classify_comments(comments: List[Comment]):
    comment_bodies = [comment.body_html for comment in comments]
    predictions = classifier.predict(comment_bodies)
    return [ClassifiedComment(**comment.dict(), sentiment_score=score) for comment, score in zip(comments, predictions)]
