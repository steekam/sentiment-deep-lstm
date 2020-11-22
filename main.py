from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from src.sentiment_classifier import SentimentClassifier


class BaseComment(BaseModel):
    id: str
    body_html: str


class ClassifiedComment(BaseComment):
    sentiment_score: float

app = FastAPI()

classifier = SentimentClassifier()

@app.post("/classify", response_model=List[ClassifiedComment])
async def classify_comments(comments: List[BaseComment]):
    comment_bodies = [comment.body_html for comment in comments]
    predictions = classifier.predict(comment_bodies)
    return [ClassifiedComment(**comment.dict(), sentiment_score=score) for comment, score in zip(comments, predictions)]
