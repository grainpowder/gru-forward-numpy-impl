import logging
import pathlib
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import sentencepiece as spm
import tensorflow as tf
import tensorflow.keras as keras
from scipy.special import softmax
import time
from tqdm import tqdm


class TensorflowGRU:

    def __init__(self, tokenizer: spm.SentencePieceProcessor, model: keras.Sequential):
        self.tokenizer = tokenizer
        self.model = model

    def predict(self, title: str, num_predictions: int) -> List[Tuple[int, float]]:
        tokenized_title = self.tokenizer.encode(title) if title else [1]  # 1 stands for <UNK> token
        probabilities = self.model(tf.constant([tokenized_title]))
        prediction = sorted(enumerate(probabilities.numpy()[0]), key=lambda x: x[1], reverse=True)[:num_predictions]
        return prediction


class NumpyGRU:

    def __init__(self, tokenizer: spm.SentencePieceProcessor, model: keras.Sequential):
        """
        Point is to tear set of trained parameters into separate parts and re-implement forward computation with them

        embedding    : (vocab_size, embed_dim) shaped array
        input_kernel : (embed_dim, 3 * hidden_dim) shaped array
        input_bias   : (3 * hidden_dim,) shaped array
         -> Affine transformation of each embedding vector will be cached. This will reduce computation during inference

        hidden_kernel : (hidden_dim, 3 * hidden_dim) shaped array
        hidden_bias   : (3 * hidden_dim,) shaped array

        dense_kernel  : (hidden_dim, num_categories) shaped array
        dense_bias    : (hidden_dim,) shaped array
        """
        self.tokenizer = tokenizer
        embedding = model.weights[0].numpy()
        input_kernel = model.weights[1].numpy()
        input_bias = model.weights[3].numpy()[0, :]
        self.embedding_affine_cache = embedding @ input_kernel + np.outer(np.ones(embedding.shape[0]), input_bias)
        self.hidden_kernel = model.weights[2].numpy()
        self.hidden_bias = model.weights[3].numpy()[1, :]
        self.hidden_dim = self.hidden_kernel.shape[0]
        self.dense_kernel = model.weights[4].numpy()
        self.dense_bias = model.weights[5].numpy()

    def predict(self, title: str, num_predictions: int) -> List[Tuple[int, float]]:
        tokenized_title = self.tokenizer.encode(title) if title else [1]  # 1 stands for <UNK> token
        hidden = np.zeros(self.hidden_dim, dtype=float)
        for token in tokenized_title:
            hidden = self._calculate_next_hidden(self.hidden_dim, token, hidden)
        logits = (hidden * (hidden > 0)) @ self.dense_kernel + self.dense_bias
        probabilities = softmax(logits)
        prediction = [(int(index), probabilities[index]) for index in np.argsort(-logits)[:num_predictions]]
        return prediction

    def _calculate_next_hidden(self, hidden_dim: int, current_token: int, previous_hidden: np.array) -> np.array:
        transformed_embedding = self.embedding_affine_cache[current_token, :]
        transformed_hidden = previous_hidden @ self.hidden_kernel + self.hidden_bias

        update_operand = transformed_embedding[:hidden_dim] + transformed_hidden[:hidden_dim]
        reset_operand = transformed_embedding[hidden_dim:(2 * hidden_dim)] + transformed_hidden[hidden_dim:(2 * hidden_dim)]

        update_gate = sigmoid(update_operand)
        reset_gate = sigmoid(reset_operand)

        candidate_operand = transformed_embedding[(2 * hidden_dim):] + reset_gate * transformed_hidden[(2 * hidden_dim):]
        candidate_hidden = np.tanh(candidate_operand)

        return (1 - update_gate) * candidate_hidden + update_gate * previous_hidden


def evaluate(project_dir: pathlib.Path, logger: logging.Logger) -> None:
    data_dir = project_dir.joinpath("data")
    model_dir = project_dir.joinpath("models")

    logger.info("Load data, tokenizer and trained model")
    test_data = pd.read_csv(data_dir.joinpath("test_data.csv"))
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_dir.joinpath("tokenizer.model")))
    model = keras.models.load_model(filepath=model_dir.joinpath("tensorflow"))
    tf_model = TensorflowGRU(tokenizer, model)
    np_model = NumpyGRU(tokenizer, model)

    logger.info("Make prediction using each model and compare set of two predictions")
    comparison_result = compare_result_and_speed(test_data["title"].values, tf_model, np_model)

    num_titles = test_data.shape[0]
    result_corresponds = comparison_result["result_corresponds"].sum() == num_titles
    logger.info(f"Two prediction results on every one of {num_titles:,} titles are same : {result_corresponds}")

    logger.info(f"However, inference speed of numpy implementation is several times faster than original model")
    elapse_time_result = comparison_result[["title_length", "tf_elapse_ms", "np_elapse_ms"]].\
        groupby("title_length").aggregate({"tf_elapse_ms": np.median, "np_elapse_ms": np.median}).\
        reset_index().values

    header = "Length | tf elapse(ms) | np elapse(ms)"
    logger.info(header)
    logger.info("-" * len(header))
    for index in [int(value) for value in np.linspace(0, elapse_time_result.shape[0] - 1, 5)]:
        title_length, tf_elapse, np_elapse = elapse_time_result[index]
        tf_elapse = str(round(tf_elapse, 4))
        np_elapse = str(round(np_elapse, 4))
        logger.info(f"{str(int(title_length)).rjust(6)} | {tf_elapse.rjust(13)} | {np_elapse.rjust(13)}")


def compare_result_and_speed(titles: Iterable[str], tf_model: TensorflowGRU, np_model: NumpyGRU):
    result_corresponds = []
    title_length = []
    tf_elapse_ms = []
    np_elapse_ms = []
    num_predictions = 3
    for title in tqdm(titles):
        title_length.append(len(title))

        start = time.time()
        tf_prediction = tf_model.predict(title, num_predictions)
        elapsed_time = (time.time() - start) * 1000
        tf_elapse_ms.append(elapsed_time)

        start = time.time()
        np_prediction = np_model.predict(title, num_predictions)
        elapsed_time = (time.time() - start) * 1000
        np_elapse_ms.append(elapsed_time)

        tf_prediction = [pair[0] for pair in tf_prediction]
        np_prediction = [pair[0] for pair in np_prediction]
        result_corresponds.append(all(map(lambda x: x[0] == x[1], zip(tf_prediction, np_prediction))))

    return pd.DataFrame({
        "title_length": title_length,
        "tf_elapse_ms": tf_elapse_ms,
        "np_elapse_ms": np_elapse_ms,
        "result_corresponds": result_corresponds
    })


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))
