import tensorflow as tf
import numpy as np
from pathlib import Path
import re
import string
from typing import List


class SentimentClassifier:
    model: tf.keras.Sequential

    def __init__(self) -> None:
        self.model = tf.keras.models.load_model(
            f"{Path.cwd()}/sentiment_model",
            compile=False,
            custom_objects={"custom_standardization": self.custom_standardization})

    @tf.keras.utils.register_keras_serializable()
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<[^>]+>', ' ')
        stripped_punctuation = tf.strings.regex_replace(
            stripped_html, '[%s]' % re.escape(string.punctuation), '')
        stripped_newline_chars = tf.strings.regex_replace(
            stripped_punctuation, '[\\n\\t]', '')
        return tf.strings.strip(stripped_newline_chars)

    def predict(self, input: List[str]):
        predictions = self.model.predict(np.array([input]))
        return [ prediction[0] for prediction in predictions]