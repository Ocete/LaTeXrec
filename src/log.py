import logging
import os
import hashlib
import json


def get_folder_path(args):
    """
    Compute the folder path and create it using the given arguments.

    Params:
    - args: CLI arguments.
    """

    folder_name = args.log_folder_name
    if folder_name is None:
        folder_name = hashlib.sha256(
            json.dumps(vars(args)).encode('utf8')
        ).hexdigest()

    folder_path = os.path.join('/artifacts', folder_name)
    os.mkdir(folder_path)

    return folder_path


def setup_logger(name, log_file_path, level=logging.INFO):
    """
    Sets up a new (or not) logger, and returns it.

    Params:
    - name: Name of the logger.
    - log_file_path: Path to the file where the logs will be
      saved.
    """

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stderr_handler)

    return logger


def log_params(folder_path, args):
    """
    Makes a new directory for the new experiment and logs the
    parameters used in this experiment in a new file.

    Params:
    - folder_path: path to logging folder
    - args: arguments to log
    """

    # Log params
    file_path = os.path.join(folder_path, 'params.json')
    with open(file_path, 'w') as f:
        json.dump(vars(args))


def get_logger(args, log_folder, mode):
    """
    Returns the logger for the given mode.

    Params:
    - args: CLI arguments.
    - mode: If 0, logs will go to development.log.
                    If 1, logs will go to results.log.
    """

    if mode != 0 and mode != 1:
        raise ValueError('mode must be either 0 or 1')

    name = 'development' if mode == 0 else 'results'
    file_path = os.path.join(log_folder, name + '.log')

    return setup_logger(name, file_path)
