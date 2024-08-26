import numpy as np
from data.dataset_loader.utils import divide_array, LocalDataset
import os
import os.path
from torch.utils.data import Dataset
from global_config import ROOT_DIRECTORY, PROJECTS_DIRECTORY
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF


class MNISTDataset(Dataset):
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            dataset_folder_path = os.path.join(ROOT_DIRECTORY, "data", "dataset", "MNIST")

        mnist_train = datasets.MNIST(root=dataset_folder_path, train=True, download=True)
        mnist_test = datasets.MNIST(root=dataset_folder_path, train=False, download=True)

        x_train = mnist_train.train_data
        y_train = mnist_train.train_labels
        x_test = mnist_test.train_data
        y_test = mnist_test.train_labels

        x_train = torch.reshape(x_train, shape=(len(x_train), 1, 28, 28)) / 255.
        x_test = torch.reshape(x_test, shape=(len(x_test), 1, 28, 28)) / 255.

        self.imgs = torch.concat([x_train, x_test], dim=0).type(torch.float32)
        self.labels = torch.concat([y_train, y_test], dim=0)

    def apply_rotations(self, n_angles=8):
        subset_size = self.imgs.shape[0] // n_angles
        # Define the angles for each subset
        if n_angles == 8:
            angles = [0, 15, 90, 105, 180, 195, 270, 275]
        elif n_angles == 4:
            angles = [0, 90, 180, 270]

        # Apply rotations to each subset
        for i in range(n_angles):
            start = i * subset_size
            end = (i + 1) * subset_size if i < (n_angles - 1) else None
            for j in range(start, end if end is not None else self.imgs.shape[0]):
                self.imgs[j] = TF.rotate(self.imgs[j], angles[i])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


def get_rotated_mnist_8_angles_local_datasets(config):
    mnist_dataset = MNISTDataset()
    mnist_dataset.apply_rotations(n_angles=8)
    dataset_length = len(mnist_dataset)
    subarrays = np.array_split(np.arange(dataset_length),
                               config["n_clients"])  # no exception if groups aren't equal length.
    local_datasets = [LocalDataset(mnist_dataset, indices, task=config["task"]) for indices in subarrays]

    return local_datasets, [None for _ in range(config["n_clients"])]


def get_rotated_mnist_4_angles_local_datasets(config):
    mnist_dataset = MNISTDataset()
    mnist_dataset.apply_rotations(n_angles=4)
    dataset_length = len(mnist_dataset)
    subarrays = np.array_split(np.arange(dataset_length),
                               config["n_clients"])  # no exception if groups aren't equal length.
    local_datasets = [LocalDataset(mnist_dataset, indices, task=config["task"]) for indices in subarrays]

    return local_datasets, [None for _ in range(config["n_clients"])]


if __name__ == "__main__":
    # random data
    mnist_dataset = MNISTDataset()
