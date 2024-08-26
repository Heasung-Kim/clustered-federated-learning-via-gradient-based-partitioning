import torch
import argparse
from global_config import base_config, global_logger, ROOT_DIRECTORY, logging
from data.dataset_loader.utils import get_datasets
from model.utils import get_model_class
from learning_agent.central_unit import CentralUnit
from learning_agent.client import Client
from pathlib import Path
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)

# Model configuration
parser.add_argument("--n_models", type=int, default=4)
parser.add_argument("--n_clients", type=int, default=128)

# Task and Dataset
parser.add_argument("--task", type=str, default='classification', choices=['classification',
                                                                           'DORO_compression',
                                                                           'compression',
                                                                           'vqvae_compression',
                                                                           'regression'])

parser.add_argument("--dataset", type=str, default='rotated_mnist_8_angles', choices=[
                                                                            'rotated_mnist_8_angles',
                                                                            'rotated_mnist_4_angles',
                                                                            'label_permuted_cifar10',
                                                                            'dirichlet_rotated_emnist',
                                                                            'gaussian_linear_regression'])

# Model selection
parser.add_argument("--model", type=str, default='SimpleLinear', choices=['SimpleLinear',
                                                                          'MADMOConvNet',
                                                                          'LinearRegressor',
                                                                          'Resnet18'])

parser.add_argument("--random_seed", type=int, default=1)

parser.add_argument("--unknown_k", type=bool, default=False)
parser.add_argument("--warmup_epoch", type=int, default=0)

# For autoencoders
parser.add_argument("--n_codeword_floats", type=int, default=16)
parser.add_argument("--n_codeword_bits", type=int, default=256)

# miscellaneous
parser.add_argument("--use_single_model_criteria", type=bool, default=False)
parser.add_argument("--use_one_beta", type=bool, default=False)
parser.add_argument("--gap_degree", type=float, default=5)
parser.add_argument("--mixture_ratio", type=float, default=0.9)
parser.add_argument("--rho", type=float, default=0.1)  # Imbalanced dataset tests
parser.add_argument("--linear_regression_initial_value_range", type=float, default=0.8)

# Compressed gradient information
parser.add_argument("--use_reduced_G", type=bool, default=False)
parser.add_argument("--gradient_compression_ratio", type=int, default=100)
parser.add_argument("--use_decoder_gradient_only", type=bool, default=False)
parser.add_argument("--use_last_layer_only", type=bool, default=True)

# Training configuration
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--n_communication_rounds", type=int, default=20)
parser.add_argument("--clustering_period", type=int, default=2)
parser.add_argument("--clustering_termination_threshold", type=int, default=10)

# Algorithmic configuration
parser.add_argument("--algorithm", type=str, default='MADMO', choices=['CFLGP', 'IFCA', 'MADMO', 'FEDAVG'])
parser.add_argument("--local_update_epoch", type=int, default=1)
parser.add_argument("--protocol", type=str, default='gradient_averaging', choices=['gradient_averaging',
                                                                                   'model_averaging'])
parser.add_argument("--subprotocol", type=str, default='no_subprotocol', choices=['encoder_averaging',
                                                                                  'cnn_averaging',
                                                                                  'no_subprotocol'])
parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd', 'adam'])
parser.add_argument("--use_scheduler", type=bool, default=False)

# Results
result_path = os.path.join(ROOT_DIRECTORY, "results")
parser.add_argument("--results_path_name", type=str, default=result_path)
parser.add_argument("--save", type=bool, default=True)
parser.add_argument("--name", type=str, default="")
parser.add_argument("--train_dataset_ratio", type=float, default=0.7)
parser.add_argument("--evaluation_interval", type=int, default=2)

parser.add_argument("--load", type=bool, default=False)
parser.add_argument("--test", type=bool, default=False)

args = parser.parse_args()
config = base_config
config.update(vars(args))
config["name"] = (config["name"] + config["algorithm"] + "_" + config["protocol"] + "_" + config["dataset"] + "_"
                  + config["model"] + "_" + config["task"] + "_b" + str(config["batch_size"]) + "_nc"
                  + str(config["n_clients"]) + "_seed" + str(config["random_seed"]))

gpu_num = int(args.gpu)
device = torch.device("cuda:" + str(gpu_num) if torch.cuda.is_available() and gpu_num > -1 else "cpu")

log_path = os.path.join(config["results_path_name"], config["name"], "logs")
Path(log_path).mkdir(parents=True, exist_ok=True)
fileh = logging.FileHandler(os.path.join(log_path, "log.txt"), 'a')
global_logger.addHandler(fileh)

Path(os.path.join(config["results_path_name"], config["name"])).mkdir(parents=True, exist_ok=True)
with open(os.path.join(config["results_path_name"], config["name"], "config.json"), 'w') as fp:
    json.dump(config, fp, indent=4)

import numpy as np
import random

torch.manual_seed(config["random_seed"])
torch.cuda.manual_seed(config["random_seed"])
torch.backends.cudnn.deterministic = True
np.random.seed(config["random_seed"])
random.seed(config["random_seed"])

local_datasets, test_datasets = get_datasets(config=config)
model_class = get_model_class(config=config)

# Optimizer
if config["optimizer"] == "adam":
    optimizer = lambda x: torch.optim.Adam(x, lr=config["learning_rate"])
elif config["optimizer"] == "sgd":
    optimizer = lambda x: torch.optim.SGD(x, lr=config["learning_rate"])
else:
    raise NotImplementedError

# Client
clients = [Client(model_class=model_class,
                  optimizer=optimizer,
                  local_dataset=local_dataset,
                  test_dataset=test_datasets[cnt],
                  use_scheduler=config["use_scheduler"],
                  batch_size=config["batch_size"],
                  train_dataset_ratio=config["train_dataset_ratio"],
                  local_update_epoch=config["local_update_epoch"],
                  task=config["task"],
                  device=device)
           for cnt, local_dataset in enumerate(local_datasets)]

# Central Unit
central_unit = CentralUnit(model_class=model_class,
                           clients=clients,
                           name=config["name"],
                           clustering_period=config["clustering_period"],
                           n_communication_rounds=config["n_communication_rounds"],
                           learning_rate=config["learning_rate"],
                           n_models=config["n_models"],
                           results_path_name=config["results_path_name"],
                           load=config["load"],
                           optimizer_name=config["optimizer"],
                           algorithm=config["algorithm"],
                           evaluation_interval=config["evaluation_interval"],
                           protocol=config["protocol"],
                           subprotocol=config["subprotocol"],
                           task=config["task"],
                           use_reduced_G=config["use_reduced_G"],
                           unknown_k=config["unknown_k"],
                           warmup_epoch=config["warmup_epoch"],
                           clustering_termination_threshold=config["clustering_termination_threshold"],
                           device=device)

if config["test"] == True:
    central_unit.test()
else:
    central_unit.train()
