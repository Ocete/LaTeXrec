import logging
import os

configured = False

def log(msg):
	"""
	Logs a single message, both to the standard output
	and the previously configured log file.

	Params:
    - msg: message to be logged.
	"""

	if not configured:
		raise RuntimeError('Trying to log without a previous init')

	print(msg)
	logging.info(msg)


def get_folder_path(args):
	"""
	Compute the folder path using the given arguments

	Params:
	- args: CLI arguments.
	"""

	folder_name = ''
	try:
		folder_name =  args.log_folder_name
	except:
		# TODO: implement the hash depending on the args
		folder_name = 'logging folder'

	folder_path = os.path.join('/artifacts', folder_name)
	if os.path.exists(folder_path):
		raise ValueError('Folder {} already exists'.format(
				folder_path))

	return folder_path


def set_logging_config(file_path):
	"""
	Configures where to log from now on

	Params:
	- file_path: file path where logs will be saved.
	"""

	logging.basicConfig(filename=file_path,
						encoding='utf-8', 
						level=logging.INFO)


def log_params(args):
	"""
	Makes a new directory for the new experiment and logs the
	parameters used in this experiment in a new file.

	Params:
	- args: CLI arguments.
	"""

	folder_path = get_folder_path(args)
	os.mkdir(folder_path)

	# First logging to save the parameters of the experiment
	file_path = os.path.join(folder_path + 'params.log')
	set_logging_config(file_path)
	log('PARAMS:\n{}'.format(args))


def set_logging(args, mode):
	"""
	Initializes the logging so from now, all logs go to the
	either the development or the results file.

	Params:
	- args: CLI arguments.
	- mode: If 0, logs go to development.log.
			If 1, logs fo to results.log.
	"""

	if mode != 0 and mode != 1:
		raise ValueError('mode must be either 0 or 1')

	configured = True

	folder_path = get_folder_path(args)
	file_name = 'development.log' if mode == 0 else 'results.log'
	file_path = os.path.join(folder_path , file_name)

	set_logging_config(file_path)

	if mode == 0:
		log('DEVELOPMENT:')
	else:
		log('FINAL RESULTS:')
