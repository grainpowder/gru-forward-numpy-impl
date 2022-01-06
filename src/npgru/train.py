import logging
import pathlib
from typing import List, Iterable

import numpy as np
import pandas as pd
import sentencepiece as spm
import tensorflow as tf
import tensorflow.keras as keras
from gensim.models import Word2Vec
import shutil


def train(
        project_dir: pathlib.Path,
        logger: logging.Logger,
        vocab_size: int,
        embed_dim: int,
        num_epochs: int,
        batch_size: int,
        num_predict: int
) -> None:
    data_dir = project_dir.joinpath("data")
    model_dir = project_dir.joinpath("models")

    logger.info("Training tokenizer under unigram assumption")
    tokenizer = train_tokenizer(data_dir, model_dir, vocab_size)

    logger.info("Training word embeddings using skip-gram algorithm")
    titles = []
    with open(data_dir.joinpath("processed_titles.txt"), "r") as file:
        for title in file:
            titles.append(title)
    tokenized_titles = tokenize_to_array(tokenizer, titles)
    embeddings = train_embeddings(tokenized_titles, vocab_size, embed_dim)

    logger.info("Initializing evaluation dataset and classification model")
    feed_data = pd.read_csv(data_dir.joinpath("feed_data.csv"))
    num_categories = feed_data["category"].max() + 1
    eval_data = pd.read_csv(data_dir.joinpath("eval_data.csv"))
    eval_dataset = initialize_dataset(tokenizer, eval_data, False, batch_size)
    model = initialize_classifier(num_categories, embeddings)

    logger.info("Train classification model")
    optimizer = keras.optimizers.Adam()
    loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    top_k_precision = keras.metrics.SparseTopKCategoricalAccuracy(k=num_predict)
    num_batches = feed_data.shape[0] // batch_size
    verbose_unit = num_batches // 5
    for epoch_index in range(num_epochs):
        logger.info(f"########## Epoch [{epoch_index + 1} / {num_epochs}] ##########")
        feed_dataset = initialize_dataset(tokenizer, feed_data, True, batch_size)
        batch_losses = []

        for batch_index, batch in enumerate(feed_dataset):
            tensor_titles, tensor_categories = batch
            with tf.GradientTape() as tape:
                probabilities = model(tensor_titles)
                loss = loss_function(tensor_categories, probabilities)
            batch_losses.append(loss.numpy().mean())
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if batch_index % verbose_unit == 0:
                avg_batch_loss = np.mean(batch_losses)
                logger.info(f"Batch [{batch_index:>3}/{num_batches}] | Average loss : {avg_batch_loss:.4f}")
        logger.info(f"Finished training process for epoch {epoch_index + 1}")

        top_k_precision.reset_state()
        for batch in eval_dataset:
            tensor_titles, tensor_categories = batch
            tensor_categories = tensor_categories[:, tf.newaxis]
            probabilities = model(tensor_titles)
            top_k_precision.update_state(tensor_categories, probabilities)
        calculated_metric = top_k_precision.result().numpy()
        logger.info(f"Top-3-precision on evaluation dataset : {calculated_metric:.4f}")

    logger.info("Save trained classifier")
    classifier_dir = model_dir.joinpath("tensorflow")
    classifier_dir.mkdir(parents=True, exist_ok=True)
    keras.models.save_model(model, str(classifier_dir))
    shutil.make_archive(classifier_dir, "zip", classifier_dir)


def train_tokenizer(data_dir: pathlib.Path, model_dir: pathlib.Path, vocab_size: int) -> spm.SentencePieceProcessor:
    spm.SentencePieceTrainer.train(
        input=str(data_dir.joinpath("processed_titles.txt")),
        model_prefix=str(model_dir.joinpath("tokenizer")),
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="unigram",
        add_dummy_prefix=False,
        minloglevel=2,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_dir.joinpath("tokenizer.model")))
    return tokenizer


def tokenize_to_array(tokenizer: spm.SentencePieceProcessor, titles: List[str]) -> List[List[int]]:
    tokenized_titles = []
    for title in titles:
        tokenized_titles.append(tokenizer.encode(title))
    return tokenized_titles


def train_embeddings(tokenized_titles: List[List[int]], vocab_size: int, embed_dim: int) -> np.array:
    model = Word2Vec(sentences=tokenized_titles, vector_size=embed_dim, min_count=5, sg=1)
    embeddings = np.random.random((vocab_size, embed_dim)) * 2 - 1  # initialize from U(-1, 1) distribution
    for token_index, embed_index in model.wv.key_to_index.items():
        embeddings[token_index, :] = model.wv.vectors[embed_index, :]
    return embeddings


def tokenize_to_tensor(tokenizer: spm.SentencePieceProcessor, titles: Iterable[str], subword: bool) -> tf.RaggedTensor:
    tokenized_titles = []
    for title in titles:
        if isinstance(title, float):
            tokenized_title = [1]
        elif subword:
            tokenized_title = tokenizer.SampleEncodeAsIds(title, 10, 0.2)
        else:
            tokenized_title = tokenizer.encode(title)
        tokenized_title = list(filter(lambda token_index: token_index != 1, tokenized_title))  # 1 stands for <UNK>
        result = [1] if len(tokenized_title) == 0 else tokenized_title
        tokenized_titles.append(result)
    return tf.ragged.constant(tokenized_titles)


def initialize_dataset(
        tokenizer: spm.SentencePieceProcessor,
        data: pd.DataFrame,
        is_feed: bool,
        batch_size: int
) -> tf.data.Dataset:
    tensor_titles = tokenize_to_tensor(tokenizer, data["title"].values, is_feed)
    tensor_category = tf.cast(list(data["category"].values), tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((tensor_titles, tensor_category)). \
        shuffle(data.shape[0]). \
        batch(batch_size)
    return dataset


def initialize_classifier(num_categories: int, embeddings: np.array) -> keras.Sequential:
    vocab_size, embed_dim = embeddings.shape
    embedding_layer = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        embeddings_initializer=keras.initializers.constant(embeddings),
        trainable=False,
        name="embedding_layer"
    )
    gru_cell = keras.layers.GRU(
        units=embed_dim * 2,
        return_sequences=False,
        name="gru_cell"
    )
    relu_activation = keras.layers.Activation(
        activation="relu",
        name="relu_activation"
    )
    dense_layer = keras.layers.Dense(
        units=num_categories,
        activation="softmax",
        input_shape=(embed_dim * 2,),
        name="dense_layer"
    )
    model = keras.Sequential(
        layers=[embedding_layer, gru_cell, relu_activation, dense_layer],
        name="classification_model"
    )
    return model
