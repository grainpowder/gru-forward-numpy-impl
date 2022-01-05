from typer import Typer

import util
from npgru.preprocess import preprocess

app = Typer()
project_dir = util.get_project_dir()


@app.command("preprocess")
def run_preprocess():
    logger = util.make_logger(project_dir, "preprocess")
    preprocess(project_dir, logger)


@app.command("train")
def run_train():
    pass


if __name__ == '__main__':
    app()
