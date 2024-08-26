# References:
#
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torchvision.transforms as transforms
import numpy as np
import torchvision
import torch
from data.dataset_loader.utils import divide_array, LocalDataset
import os
import os.path
from torch.utils.data import Dataset
from global_config import ROOT_DIRECTORY, PROJECTS_DIRECTORY
from pathlib import Path


class CIFAR10Dataset(Dataset):
    def __init__(self, dataset_path=None, type='train'):
        if dataset_path is None:
            dataset_path = os.path.join(ROOT_DIRECTORY,  "data", "dataset", "CIFAR10")
            Path(dataset_path).mkdir(parents=True, exist_ok=True)

        dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=type == 'train',
                                               download=True)
        self.dataset_length = len(dataset)
        self.imgs = torch.from_numpy(np.transpose(dataset.data / 255., (0, 3, 1, 2)).astype(np.float32))
        self.labels = torch.from_numpy(np.array(dataset.targets)).type(torch.LongTensor)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


def get_label_permuted_cifar10_local_datasets(config):
    cifar_dataset = CIFAR10Dataset()
    dataset_length = cifar_dataset.dataset_length
    subarrays = []

    N_CLUSTERS = 2  # config['n_models']
    N_CLIENT = config['n_clients']
    labels = cifar_dataset.labels.detach().numpy()

    indices = np.arange(dataset_length)
    np.random.shuffle(indices)
    cluster_1_indices, cluster_2_indices = np.split(indices, 2)
    entire_cluster_subarrays = [cluster_1_indices, cluster_2_indices]

    bt = torch.zeros(dataset_length, dtype=torch.bool)
    bt[cluster_1_indices] = True

    new_label = torch.where(bt, (cifar_dataset.labels + 5) % 10, cifar_dataset.labels)
    cifar_dataset.labels = new_label

    for k in range(N_CLUSTERS):
        cluster_subarrays = np.array_split(entire_cluster_subarrays[k], N_CLIENT // N_CLUSTERS)
        for c in range(len(cluster_subarrays)):
            subarrays.append(cluster_subarrays[c])
    local_datasets = [LocalDataset(cifar_dataset, indices, task=config["task"]) for indices in subarrays]

    for i, local_dataset in enumerate(local_datasets):
        local_dataset.transformation_function = transforms.Compose(
            [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return local_datasets, [None for _ in range(config["n_clients"])]


if __name__ == "__main__":
    # random data
    dataset = CIFAR10Dataset()
