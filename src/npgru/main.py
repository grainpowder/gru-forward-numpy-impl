from typer import Typer
import typer
import util
from npgru.preprocess import preprocess
from npgru.train import train
from npgru.evaluate import evaluate
from npgru.upload import upload

app = Typer()
project_dir = util.get_project_dir()


@app.command("preprocess")
def run_preprocess():
    logger = util.make_logger(project_dir, "preprocess")
    preprocess(project_dir, logger)


@app.command("train")
def run_train(
        vocab_size: int = typer.Option(15000, "-v", "--vocab-size"),
        embed_dim: int = typer.Option(128, "-d", "--embed-dim"),
        num_epochs: int = typer.Option(3, "-e", "--num-epochs"),
        batch_size: int = typer.Option(128, "-b", "--batch-size"),
        num_predict: int = typer.Option(3, "-k", "--top-k-predict"),
):
    logger = util.make_logger(project_dir, "train")
    train(project_dir, logger, vocab_size, embed_dim, num_epochs, batch_size, num_predict)


@app.command("evaluate")
def run_evaluate():
    logger = util.make_logger(project_dir, "evaluate")
    evaluate(project_dir, logger)


@app.command("upload")
def run_upload():
    logger = util.make_logger(project_dir, "upload")
    upload(project_dir, logger)


if __name__ == '__main__':
    model_dir = project_dir.joinpath("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    app()
