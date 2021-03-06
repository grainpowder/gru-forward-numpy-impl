import os

import pandas as pd

import npgru.preprocess as preprocess


def test_get_file_names():
    os.environ["ZIPFILE_NAME"] = "test.zip"
    os.environ["FILE_NAME"] = "test.csv"
    zipfile_name, file_name = preprocess.get_file_names()
    assert zipfile_name == "test.zip"
    assert file_name == "test.csv"


def test_process_titles():
    test_data = pd.DataFrame({
        "title": [
            "How to Write Clean Code (in Python)",
            "βοΈ Copy and π Paste π",
            "μΈγγλνλ€. κ°μ΄μ λΉμκ° λ μμγγγ κ½νλ€."
        ],
        "target": [
            "how to write clean code in python",
            "copy and paste",
            "μΈ λνλ€ κ°μ΄μ λΉμκ° λ μμ κ½νλ€"
        ]
    })
    result = preprocess.process_titles(test_data)
    assert all(result == test_data["target"].values)
