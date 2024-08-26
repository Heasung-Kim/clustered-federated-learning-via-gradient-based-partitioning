import numpy as np
import torch
from data.dataset_loader.utils import divide_array, LocalDataset
import os
import os.path
from torch.utils.data import Dataset
from global_config import ROOT_DIRECTORY, PROJECTS_DIRECTORY
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='whitegrid', font_scale=2.)
sns.set_style({'font.family': 'Times New Roman'})

fig, axes = plt.subplots(1, 1, figsize=(4, 8))
plt.tight_layout()


class LinearRegressionDataset(Dataset):
    def __init__(self, config, n_samples_per_cluster, dataset_path=None):
        dataset_folder_path = os.path.join(PROJECTS_DIRECTORY, "dataset", "gaussian_linear_regression")
        np.random.seed(config['random_seed'])
        num_clusters = 3

        # np.random.seed(42)
        n_samples = n_samples_per_cluster
        x_min, x_max = 0, 1

        noise_std = 0.2

        gap_degree = config["gap_degree"]

        delta_theta = 2 * np.pi / 360 * gap_degree

        center_theta = 0  # (np.pi / 2) / 2

        theta1 = center_theta - delta_theta  # rad
        x1 = np.random.uniform(low=x_min, high=np.cos(theta1), size=n_samples)
        noise = np.random.normal(loc=0, scale=noise_std, size=n_samples)
        y1 = x1 * np.tan(theta1) + noise

        theta2 = center_theta
        x2 = np.random.uniform(low=x_min, high=np.cos(theta2), size=n_samples)
        noise = np.random.normal(loc=0, scale=noise_std, size=n_samples)
        y2 = x2 * np.tan(theta2) + noise

        theta3 = center_theta + delta_theta
        x3 = np.random.uniform(low=x_min, high=np.cos(theta3), size=n_samples)
        noise = np.random.normal(loc=0, scale=noise_std, size=n_samples)
        y3 = x3 * np.tan(theta3) + noise

        # Plot the data
        PLOT = False
        if PLOT:
            plt.scatter(x1, y1, s=10, c='r', alpha=0.1, marker='d', label="D1")
            x = np.linspace(0, np.cos(theta1), 100)  # generate 100 x-values between 0 and cos(angle)
            y = x * np.tan(theta1)  # calculate the corresponding y-values for each x-value based on the angle
            # plot the line
            plt.plot(x, y, '--', c='r')

            plt.scatter(x2, y2, s=10, c='g', alpha=0.1, marker='s', label="D2")

            x = np.linspace(0, np.cos(theta2), 100)  # generate 100 x-values between 0 and cos(angle)
            y = x * np.tan(theta2)  # calculate the corresponding y-values for each x-value based on the angle
            # plot the line
            plt.plot(x, y, '--', c='g')

            plt.scatter(x3, y3, s=10, c='b', alpha=0.1, marker='o', label="D3")

            x = np.linspace(0, np.cos(theta3), 100)  # generate 100 x-values between 0 and cos(angle)
            y = x * np.tan(theta3)  # calculate the corresponding y-values for each x-value based on the angle
            # plot the line
            plt.plot(x, y, '--', c='b')
            # plt.title("Synthetic Dataset")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.ylim(-1.0, 1.0)
            plt.gca().set_aspect('equal')
            # plt.legend()
            plt.tight_layout()
            plt.show()

        self.imgs = torch.from_numpy(np.reshape(np.concatenate([x1, x2, x3]), newshape=(-1, 1)).astype(np.float32))
        self.labels = torch.from_numpy(np.reshape(np.concatenate([y1, y2, y3]), newshape=(-1, 1)).astype(np.float32))

        print("Dataset generated.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


def get_gaussian_linear_regression_local_datasets(config):
    dataset = LinearRegressionDataset(config, n_samples_per_cluster=262144)
    dataset_length = len(dataset)
    indices = np.arange(dataset_length)
    subarrays = divide_array(indices, n_clients=config["n_clients"])
    local_datasets = [LocalDataset(dataset, indices, task=config["task"]) for indices in subarrays]

    test_local_datasets = LinearRegressionDataset(config, n_samples_per_cluster=131072)
    test_dataset_length = len(test_local_datasets)
    test_indices = np.arange(test_dataset_length)
    test_subarrays = divide_array(test_indices, n_clients=config["n_clients"])
    test_local_datasets = [LocalDataset(test_local_datasets, indices, task=config["task"]) for indices in
                           test_subarrays]

    return local_datasets, test_local_datasets


if __name__ == "__main__":
    # random data
    config = {
        "gap_degree": 40,
        "random_seed": 10
    }
    mnist_dataset = LinearRegressionDataset(config=config, n_samples_per_cluster=1000)
