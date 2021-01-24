import logging
import os
import hashlib
import json

def log(logger, msg):
	"""
	Logs a single message, both to the standard output
	and the previously configured log file.

	Params:
	- logger: logger, where to log the message.
    - msg: message to be logged.
	"""

	print(msg)
	logger.info(msg)


def get_folder_path(args):
	"""
	Compute the folder path using the given arguments

	Params:
	- args: CLI arguments.
	"""

	folder_name =  args.log_folder_name
	if folder_name is None:
		folder_name = hashlib.sha256(
							json.dumps(vars(args)).encode('utf8')
						).hexdigest()

	folder_path = os.path.join('/artifacts', folder_name)

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
    handler = logging.FileHandler(log_file_path) 
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def log_params(args):
	"""
	Makes a new directory for the new experiment and logs the
	parameters used in this experiment in a new file.

	Params:
	- args: CLI arguments.
	"""

	folder_path = get_folder_path(args)

	# Create new folder for all the logs of this experiment
	if os.path.exists(folder_path):
		raise ValueError('Folder {} already exists'.format(
				folder_path))
	os.mkdir(folder_path)

	# Create the params logger
	name = 'params'
	file_path = os.path.join(folder_path, name + '.log')
	file_path = os.path.join(folder_path, 'params.log')
	logger = setup_logger(name, file_path)

	# First logging to save the parameters of the experiment
	log(logger, 'PARAMS:\n{}'.format(json.dumps(vars(args))))


def get_logger(args, mode):
	"""
	Returns the logger for the given mode.

	Params:
	- args: CLI arguments.
	- mode: If 0, logs will go to development.log.
			If 1, logs will go to results.log.
	"""

	if mode != 0 and mode != 1:
		raise ValueError('mode must be either 0 or 1')

	folder_path = get_folder_path(args)
	name = 'development' if mode == 0 else 'results'
	file_path = os.path.join(folder_path, name + '.log')

	return setup_logger(name, file_path)
