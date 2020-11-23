# Sentiment Classifier API
Simple API that exposes an endpoint to predict the sentiment of a list of texts. The use case for this project is to predict a list of comments from dev.to articles provided by a separate client.

## Installing dependencies
The project was created using poetry. If you have poetry installed run:
```bash
poetry install
```

## Running the server
`uvicorn` server is among the dependencies. To start the server API run:
```bash
uvicorn app.main:app --reload
```

## Tests
Simply run:
```bash
poetry run python -m pytest
```

## Contributors
[Kamau Wanyee](https://github.com/steekam)