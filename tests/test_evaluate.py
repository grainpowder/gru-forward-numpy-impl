import os
import re

import numpy as np
import sentencepiece as spm
import tensorflow.keras as keras

from npgru.evaluate import TensorflowGRU, NumpyGRU
from npgru.util import get_project_dir

project_dir = get_project_dir()
model_dir = project_dir.joinpath("models")
model = keras.models.load_model(model_dir.joinpath("tensorflow"))
tokenizer = spm.SentencePieceProcessor(model_file=str(model_dir.joinpath("tokenizer.model")))
title = "A Chat with Andrew on MLOps: From Model-centric to Data-centric AI"
processed_title = re.sub("[^가-힣a-z0-9 ]", " ", title.lower())


def test_model_dir_exists():
    assert "models" in os.listdir(project_dir), f"{model_dir} directory does not exists"


def test_evaluate_executed():
    assert "tensorflow" in os.listdir(model_dir), f"Tensorflow model not saved in {str(model_dir)}"


def test_tensorflow_gru_predict():
    tf_gru = TensorflowGRU(tokenizer, model)
    tf_pred = tf_gru.predict(processed_title, 2)
    assert all([isinstance(tf_pred[0][0], int), isinstance(tf_pred[0][1], np.float32)]), f"tf_pred[0] : {tf_pred[0]}"
    assert all([isinstance(tf_pred[1][0], int), isinstance(tf_pred[1][1], np.float32)]), f"tf_pred[1] : {tf_pred[1]}"


def test_numpy_gru_predict():
    np_gru = NumpyGRU(tokenizer, model)
    np_pred = np_gru.predict(processed_title, 2)
    assert all([isinstance(np_pred[1][0], int), isinstance(np_pred[1][1], np.float64)]), f"np_pred[0] : {np_pred[1]}"
    assert all([isinstance(np_pred[0][0], int), isinstance(np_pred[0][1], np.float64)]), f"np_pred[1] : {np_pred[0]}"
