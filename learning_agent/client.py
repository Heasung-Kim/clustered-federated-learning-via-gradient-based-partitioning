import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from learning_agent.system_functions import get_vectorized_gradient, copy_weight, copy_value, train_model, test_model, \
    get_model_difference, forward_model
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class Client:
    def __init__(self, model_class, optimizer, local_dataset, test_dataset, use_scheduler, batch_size,
                 train_dataset_ratio, local_update_epoch, task, device):
        super(Client, self).__init__()
        self.local_model = model_class().to(device)  # call constructor
        self.local_model_weights = {key: value for key, value in self.local_model.named_parameters()}
        self.local_optimizer = optimizer(self.local_model.parameters())
        self.scheduler = None
        self.use_scheduler = use_scheduler
        if self.use_scheduler is True:
            self.scheduler = CosineAnnealingWarmRestarts(self.local_optimizer, 125000, 1, eta_min=1e-4)
        self.local_dataset = local_dataset
        self.device = device
        self.batch_size = batch_size
        self.train_dataset_ratio = train_dataset_ratio
        self.local_update_epoch = local_update_epoch
        self.task = task
        self.data = local_dataset
        if test_dataset is None:
            # print("No specified test dataset is detected. Splitting train dataset...")
            n_train_data = int(len(self.data) * self.train_dataset_ratio)
            n_test_data = len(self.data) - n_train_data
            train_dataset, self.test_dataset = torch.utils.data.random_split(self.data, [n_train_data, n_test_data])
        else:
            train_dataset = self.data
            self.test_dataset = test_dataset
        self.train_dataset_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_dataset_variance = self.get_train_dataset_variance()
        self.test_dataset_loader = DataLoader(self.test_dataset, batch_size=int(np.minimum(self.batch_size * 10, 500)),
                                              shuffle=False)

        # Storage
        self.model_size_storage = {key: torch.zeros_like(value) for key, value in self.local_model.named_parameters()}
        self.n_params_per_model = sum(p.numel() for p in self.local_model.parameters())

    def weight_reset(self):
        self.local_model_weights = {key: value for key, value in self.local_model.named_parameters()}

    def get_train_dataset_variance(self):
        variance = 0.
        cnt = 0
        for images, y in self.train_dataset_loader:
            cnt += 1
        variance /= cnt
        return variance

    def compute_gradient(self, model, weights, device):
        """
        Query Type 1: gradient averaging
        """
        copy_weight(target=self.local_model_weights, source=weights)

        x, y = next(iter(self.train_dataset_loader))
        x, y = x.to(device), y.to(device)
        if self.task == "classification":
            loss = torch.nn.CrossEntropyLoss()(model(x), y)
        elif self.task == "regression":
            loss = torch.nn.MSELoss()(model(x), y)
        elif self.task == "compression":
            loss = torch.nn.MSELoss()(model(x), x)
        else:
            raise NotImplementedError
        grad = torch.autograd.grad(loss, model.parameters(), allow_unused=True,
                                   retain_graph=True)

        copy_value(target=self.model_size_storage, source=grad)
        return grad

    def compute_model_gap_by_local_update(self, model, weights, device):
        # Query Type 2 for model averaging
        # weight backup
        original_weights = weights
        copy_weight(target=self.local_model_weights, source=original_weights)
        # train temp model
        train_model(model=self.local_model,
                    data_loader=self.train_dataset_loader,
                    optimizer=self.local_optimizer,
                    scheduler=self.scheduler,
                    epochs=self.local_update_epoch,
                    task=self.task,
                    device=device,
                    train_data_variance=self.train_dataset_variance)
        # self.model_size_storage will contain the model gap, i.e., $\Delta \theta$.
        get_model_difference(self.model_size_storage, self.local_model_weights, original_weights)
        return self.model_size_storage

    def local_update(self, device, single_batch=False):
        # train temp model
        train_model(model=self.local_model,
                    data_loader=self.train_dataset_loader,
                    optimizer=self.local_optimizer,
                    scheduler=self.scheduler,
                    epochs=self.local_update_epoch,
                    task=self.task,
                    device=device, single_batch=single_batch)
        return self.model_size_storage

    def forward_propagation_model(self, device, single_batch=False):
        forward_model(model=self.local_model,
                      data_loader=self.train_dataset_loader,
                      optimizer=self.local_optimizer,
                      epochs=self.local_update_epoch,
                      task=self.task,
                      device=device, single_batch=single_batch)
        return self.model_size_storage

    # Performance evaluation query
    def performance_evaluation(self, dataset_type, device, model=None, single_batch=False):
        if model is None:
            model = self.local_model
        if dataset_type == "train":
            data_loader = self.train_dataset_loader
        elif dataset_type == "test":
            data_loader = self.test_dataset_loader
        else:
            raise ValueError
        task = self.task

        result, info = test_model(model, data_loader, task, device, single_batch=single_batch)
        return result, info

    def output_analysis(self, dataset_type, device, model=None, single_batch=False):
        from learning_agent.system_functions import result_analysis
        if model is None:
            model = self.local_model
        if dataset_type == "train":
            data_loader = self.train_dataset_loader
        elif dataset_type == "test":
            data_loader = self.test_dataset_loader
        else:
            raise ValueError
        task = self.task
        result, info = result_analysis(model, data_loader, task, device, single_batch=single_batch)
        return result, info

    def synchronize_with_central_unit_model(self, model, weights):
        # weight backup
        original_weights = weights
        copy_weight(target=self.local_model_weights, source=original_weights)
        return
