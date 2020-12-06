import uvicorn
from typing import List
from fastapi import FastAPI, HTTPException
from app.schemas import Comment, ClassifiedComment
from app.sentiment_classifier import SentimentClassifier

app = FastAPI()

classifier = SentimentClassifier()


@app.post("/classify", response_model=List[ClassifiedComment])
async def classify_comments(comments: List[Comment]):
    if not comments:
        raise HTTPException(422, "No comments provided")

    comment_bodies = [comment.body_html for comment in comments]
    predictions = classifier.predict(comment_bodies)
    return [ClassifiedComment(**comment.dict(), sentiment_score=score) for comment, score in zip(comments, predictions)]


if __name__ == '__main__':
    uvicorn.run(app, port=8080)
