import numpy as np
from torch.utils.data import Subset
from torch.utils.data import Dataset


def get_datasets(config):
    """

    info can be None. A specific test dataset can be returned through the second output element.

    :param config:
    :return: (A List of main dataset, A List of test dataset)
    """
    dataset_name = config["dataset"]

    if dataset_name == "rotated_mnist_4_angles":
        from data.dataset_loader.rotated_mnist_loader import get_rotated_mnist_4_angles_local_datasets
        return get_rotated_mnist_4_angles_local_datasets(config=config)

    elif dataset_name == "rotated_mnist_8_angles":
        from data.dataset_loader.rotated_mnist_loader import get_rotated_mnist_8_angles_local_datasets
        return get_rotated_mnist_8_angles_local_datasets(config=config)

    elif dataset_name == "dirichlet_rotated_emnist":
        from data.dataset_loader.dirichlet_rotated_emnist_loader import get_dirichlet_rotated_EMNIST_local_datasets
        return get_dirichlet_rotated_EMNIST_local_datasets(config=config)

    elif dataset_name == "gaussian_linear_regression":
        from data.dataset_loader.gaussian_linear_regression_loader import get_gaussian_linear_regression_local_datasets
        return get_gaussian_linear_regression_local_datasets(config=config)

    elif dataset_name == "label_permuted_cifar10":
        from data.dataset_loader.label_permuted_cifar10_loader import get_label_permuted_cifar10_local_datasets
        return get_label_permuted_cifar10_local_datasets(config=config)


def divide_array(labels, n_clients):
    length = len(labels)
    quotient, remainder = divmod(length, n_clients)
    start = 0
    sub_arrays = []
    for i in range(n_clients):
        end = start + quotient
        if i < remainder:
            end += 1
        sub_arrays.append(labels[start:end])
        start = end
    return sub_arrays


from data.visualization_utils import visualize_mnist


class LocalDataset(Subset):
    def __init__(self, dataset, indices, transformation_function=None, task="classification"):
        super().__init__(dataset, indices)
        self.transformation_function = transformation_function
        self.transformation_function_1 = None
        self.transformation_function_2 = None
        self.USE_MIXTURE = False
        self.MIXTURE_RATIO = None

        if type(transformation_function) is dict:
            self.transformation_function_1 = transformation_function["transformation_function_1"]
            self.transformation_function_2 = transformation_function["transformation_function_2"]
            self.USE_MIXTURE = True
            self.MIXTURE_RATIO = transformation_function["MIXTURE_RATIO"]

        self.task = task

    def __getitem__(self, idx):
        if self.task == "classification" or self.task == "regression":
            image, label = self.dataset[self.indices[idx]]
        elif "compression" in self.task:
            image, label = self.dataset[self.indices[idx]]
            label = idx
        else:
            raise NotImplementedError

        if self.transformation_function:
            if self.USE_MIXTURE is True:
                random_value = np.random.choice([0, 2], 1, p=[self.MIXTURE_RATIO, 1 - self.MIXTURE_RATIO])
                if random_value < 1:
                    image = self.transformation_function_2(image)
                else:
                    image = self.transformation_function_1(image)
            else:
                image = self.transformation_function(image)

        return image, label
