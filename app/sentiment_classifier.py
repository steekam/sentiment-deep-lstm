import tensorflow as tf
import tensorflow_text as text
import numpy as np
import re
import string
from typing import List
from config import BASE_PATH

tf.get_logger().setLevel('ERROR')


class SentimentClassifier:
    model: tf.keras.Model

    def __init__(self) -> None:
        tf.get_logger().setLevel('ERROR')
        self.model = tf.keras.models.load_model(
            f"{BASE_PATH}/small_bert_L6_H128_A2",
            compile=False)

    def custom_standardization(self, input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<[^>]+>', ' ')
        stripped_punctuation = tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')
        stripped_newline_chars = tf.strings.regex_replace(stripped_punctuation, '[\\n\\t]', '')
        return tf.strings.strip(stripped_newline_chars).numpy()

    def predict(self, input: List[str]):
        input = list(map(lambda text: self.custom_standardization(text), input))
        predictions = self.model.predict(np.array(input))
        return [prediction[0] for prediction in predictions]
