import logging
import os
import pathlib
import zipfile
from typing import Iterable, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv


def preprocess(project_dir: pathlib.Path, logger: logging.Logger) -> None:
    load_dotenv()

    logger.info("Load dataset for preprocessing")
    data_dir = project_dir.joinpath("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(data_dir)

    logger.info("Preprocess article titles")
    processed_titles = process_titles(dataset)

    logger.info("Encode category into indices")
    index2category = dict(enumerate(dataset["category"].unique()))
    category2index = dict([(category, index) for index, category in index2category.items()])
    encoded_category = [category2index[category] for category in dataset["category"].values]
    processed_dataset = pd.DataFrame({"category": encoded_category, "title": processed_titles})

    logger.info("Split dataset into feed, evaluation, test sets")
    np.random.seed(1234)
    test_dataset = processed_dataset.sample(frac=0.1, replace=False)
    train_dataset = processed_dataset.loc[~processed_dataset.index.isin(test_dataset.index)]
    eval_dataset = train_dataset.sample(n=test_dataset.shape[0], replace=False)
    feed_dataset = train_dataset.loc[~train_dataset.index.isin(eval_dataset.index)]

    logger.info("Save each of dataset and index map into data directory as csv format")
    index_map = pd.DataFrame(index2category.items(), columns=["index", "category"])
    feed_dataset.to_csv(data_dir.joinpath("feed_data.csv"), index=False)
    eval_dataset.to_csv(data_dir.joinpath("eval_data.csv"), index=False)
    test_dataset.to_csv(data_dir.joinpath("test_data.csv"), index=False)
    index_map.to_csv(data_dir.joinpath("index2category.csv"), index=False)

    logger.info("Save processed titles of feed data for tokenizer training")
    with open(data_dir.joinpath("processed_titles.txt"), "w") as file:
        file.writelines("\n".join(feed_dataset["title"].values))


def load_dataset(data_dir: pathlib.Path) -> pd.DataFrame:
    zipfile_name, file_name = get_file_names()
    if is_valid_zip_file_downloaded(data_dir, zipfile_name):
        unzipped_archive_dir = data_dir.joinpath(zipfile_name.replace(".zip", ""))
        with zipfile.ZipFile(data_dir.joinpath(zipfile_name), "r") as archive:
            file_zipinfo = archive.getinfo(file_name)
            archive.extract(file_zipinfo, unzipped_archive_dir)
    else:
        raise FileNotFoundError(f"{zipfile_name} file not found in {str(data_dir)}")
    data_path = unzipped_archive_dir.joinpath(file_name)
    return pd.read_csv(str(data_path))


def process_titles(dataset: pd.DataFrame) -> Iterable[str]:
    return dataset["title"].str.lower().str.replace("[^가-힣a-z0-9 ]", "", regex=True).values


def is_valid_zip_file_downloaded(data_dir: pathlib.Path, zipfile_name: str) -> bool:
    is_downloaded = any([filename.endswith(zipfile_name) for filename in os.listdir(data_dir)])
    is_valid = zipfile.is_zipfile(data_dir.joinpath(zipfile_name))
    return True if is_downloaded and is_valid else False


def get_file_names() -> List[str]:
    zipfile_name = os.environ.get("ZIPFILE_NAME")
    assert zipfile_name is not None, "Environment variable ZIPFILE_NAME is not defined"
    file_name = os.environ.get("FILE_NAME")
    assert file_name is not None, "Environment variable FILE_NAME is not defined"
    return [zipfile_name, file_name]
