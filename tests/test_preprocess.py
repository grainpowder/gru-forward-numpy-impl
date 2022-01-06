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
            "✂️ Copy and 📋 Paste 👌",
            "싸ㅏㅏ늘하다. 가슴에 비수가 날아와ㅏㅏㅏ 꽂힌다."
        ],
        "target": [
            "how to write clean code in python",
            " copy and  paste ",
            "싸늘하다 가슴에 비수가 날아와 꽂힌다"
        ]
    })
    result = preprocess.process_titles(test_data)
    assert all(result == test_data["target"].values)
