import logging
import os
import pathlib
import sys


def get_project_dir() -> pathlib.Path:
    """
    Utility function to return absolute directory of project directory as a pathlib.Path instance

    :return: pathlib.Path instance
    """
    root_dir = [path for path in sys.path if path.endswith("gru-forward-numpy-impl/src")]
    try:
        return pathlib.Path(root_dir[0]).parent
    except IndexError:
        print(f"Expected PYTHONPATH to be [PATH_TO_PROJECT_DIRECTORY/src], got [{os.environ.get('PYTHONPATH')}]")
        exit()


def make_logger(project_dir: pathlib.Path, log_file_name: str) -> logging.Logger:
    """
    Return a logger that both prints and saves log messages in defined format

    :param project_dir: log file will be saved in 'project_dir/logs' directory
    :param log_file_name: name of log file to be saved
    :return: logger instance
    """
    # Define log format to be used in each of handler
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | (%(funcName)s) : %(msg)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Define handler for console printouts
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # Define handler for file savings
    log_dir = project_dir.joinpath("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir.joinpath(f"{log_file_name}.txt"), mode="w")
    file_handler.setFormatter(formatter)

    # Make logger and attach each of handler to the logger
    logger = logging.getLogger("main")
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return logger
