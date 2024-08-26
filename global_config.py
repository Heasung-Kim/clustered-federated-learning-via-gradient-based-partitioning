import os

# import multiprocessing
# from multiprocessing import Lock

# define single
# Locks = [multiprocessing.Lock()]
# multiprocessing = multiprocessing
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Directory
PROJECTS_DIRECTORY = os.path.abspath(os.path.join(ROOT_DIRECTORY, os.pardir))
results_save_path = os.path.join(PROJECTS_DIRECTORY, "results")
data_storage_directory = os.path.join(PROJECTS_DIRECTORY, "dataset")

# logger
import logging

global_logger = logging.getLogger("global_logger")
global_logger.setLevel(logging.INFO)
global_logger.addHandler(logging.StreamHandler())
base_config = {
    "project_name": "CFL-GP"
}
