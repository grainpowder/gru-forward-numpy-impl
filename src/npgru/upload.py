import gzip
import logging
import os
import pathlib
import shutil
from typing import Dict

import boto3
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from dotenv import load_dotenv


def upload(project_dir: pathlib.Path, logger: logging.Logger) -> None:
    load_dotenv()
    model_dir = project_dir.joinpath("models")
    data_dir = project_dir.joinpath("data")

    logger.info("Load model to extract weights")
    model = keras.models.load_model(filepath=str(model_dir.joinpath("tensorflow")))

    logger.info("Extract weights from trained model")
    weights = extract_model_weights(model)

    logger.info("Save each set of weights as csv file")
    weight_dir = model_dir.joinpath("weights")
    weight_dir.mkdir(parents=True, exist_ok=True)
    for weight_name in weights:
        weight = weights[weight_name]
        pd.DataFrame(weight).to_csv(weight_dir.joinpath(f"{weight_name}.csv"), index=False, header=False)

    logger.info("Zip weight directory and tokenizer file")
    shutil.make_archive(weight_dir, "zip", weight_dir)
    tokenizer_name = "tokenizer.model"
    source = model_dir.joinpath(tokenizer_name)
    target = model_dir.joinpath(tokenizer_name + ".gz")
    with open(source, "rb") as source_file, gzip.open(target, "wb", compresslevel=1) as target_file:
        shutil.copyfileobj(source_file, target_file)

    logger.info("Send zipped files to S3 bucket")
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    prefix = "gru-forward-numpy"
    zipped_tokenizer_name = tokenizer_name + ".gz"
    zipped_weights_name = "weights.zip"
    zipped_data_name = os.environ.get("ZIPFILE_NAME")
    upload_to_s3(bucket_name, prefix, zipped_tokenizer_name, model_dir.joinpath(zipped_tokenizer_name))
    upload_to_s3(bucket_name, prefix, zipped_weights_name, model_dir.joinpath(zipped_weights_name))
    upload_to_s3(bucket_name, prefix, zipped_data_name, model_dir.joinpath(zipped_data_name))


def extract_model_weights(model: keras.Sequential) -> Dict[str, np.array]:
    embedding = model.weights[0].numpy()
    input_kernel = model.weights[1].numpy()
    input_bias = model.weights[3].numpy()[0, :]
    return {
        "embedding_affine": embedding @ input_kernel + np.outer(np.ones(embedding.shape[0]), input_bias),
        "hidden_kernel": model.weights[2].numpy(),
        "hidden_bias": model.weights[3].numpy()[1, :],
        "dense_kernel": model.weights[4].numpy(),
        "dense_bias": model.weights[5].numpy()
    }


def upload_to_s3(bucket_name: str, prefix: str, file_name: str, local_file_path: pathlib.Path) -> None:
    s3_client = boto3.client("s3")
    s3_key = f"{prefix}/{file_name}"
    s3_client.upload_file(str(local_file_path), bucket_name, s3_key)  # upload $1 to bucket $2 as an object with path $3
