from pydantic import BaseModel


class Comment(BaseModel):
    id: str
    body_html: str


class ClassifiedComment(Comment):
    sentiment_score: float
