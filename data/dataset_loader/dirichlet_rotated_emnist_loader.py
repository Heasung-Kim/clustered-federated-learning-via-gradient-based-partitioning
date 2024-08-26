# References:
# Code from https://github.com/felisat/clustered-federated-learning/blob/master/clustered_federated_learning.ipynb
import numpy as np
from torchvision import datasets, transforms
from data.dataset_loader.utils import divide_array, LocalDataset


def split_noniid(train_idcs, train_labels, alpha, n_clients):
    '''
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs] == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    return client_idcs


def get_dirichlet_rotated_EMNIST_local_datasets(config):
    DIRICHLET_ALPHA = 1.0
    data = datasets.EMNIST(root="./data/datasets", split="byclass", download=True)

    mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
                     'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                     'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
                     'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                     'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'], dtype='<U1')

    idcs = np.random.permutation(len(data))
    train_idcs, test_idcs = idcs[:200000], idcs[10000:20000]
    train_labels = data.train_labels.numpy()

    client_idcs = split_noniid(train_idcs, train_labels, alpha=DIRICHLET_ALPHA, n_clients=config["n_clients"])

    client_data = [LocalDataset(data, idcs, task=config["task"]) for idcs in client_idcs]
    test_data = LocalDataset(data, test_idcs, transforms.Compose([transforms.ToTensor()]))

    for i, client_datum in enumerate(client_data):
        if i < config["n_clients"] // 2:
            client_datum.transformation_function = transforms.Compose([transforms.RandomRotation((90, 90)),
                                                                       transforms.ToTensor()])
        else:
            client_datum.transformation_function = transforms.Compose([transforms.ToTensor()])

    info = [None for _ in range(config["n_clients"])]
    return client_data, info


if __name__ == "__main__":
    print("hello world!")
