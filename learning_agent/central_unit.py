import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from global_config import global_logger, logging
from learning_agent.utils import spectral_clustering_and_matching, get_num_cluster
from learning_agent.system_functions import flatten_tensor, model_averaging, gradient_averaging, copy_weight, \
    encoder_averaging, cnn_averaging
import time
from tqdm import tqdm
import os
import copy
from pathlib import Path
import pickle

np.set_printoptions(precision=5)


class CentralUnit:
    def __init__(self,
                 model_class,
                 clients,
                 name,
                 clustering_period,
                 n_communication_rounds,
                 learning_rate,
                 n_models,
                 results_path_name,
                 load,
                 optimizer_name,
                 algorithm,
                 evaluation_interval,
                 protocol,
                 subprotocol,
                 task,
                 use_reduced_G,
                 unknown_k,
                 warmup_epoch,
                 clustering_termination_threshold,
                 device):
        super(CentralUnit, self).__init__()
        self.device = device
        self.name = name
        self.clustering_period = clustering_period
        self.n_communication_rounds = n_communication_rounds
        self.learning_rate = learning_rate
        self.clients = clients
        self.n_clients = len(self.clients)
        self.estimated_cluster_ids_new = None
        self.estimated_cluster_ids = None
        self.n_models = n_models
        self.results_path_name = results_path_name
        self.load = load
        self.optimizer_name = optimizer_name
        self.algorithm = algorithm
        self.evaluation_interval = evaluation_interval
        self.protocol = protocol
        self.subprotocol = subprotocol
        self.task = task
        self.use_reduced_G = use_reduced_G
        self.gradient_compression_ratio = 100
        self.unknown_k = unknown_k
        self.warmup_epoch = warmup_epoch
        self.clustering_termination_threshold = clustering_termination_threshold

        self.reduced_gradient_profile_matrix = None
        self.singular_values = None

        # ====================================
        # Tracking tools
        # ====================================
        self.base_path = os.path.join(self.results_path_name, self.name)
        self.log_path = os.path.join(self.results_path_name, self.name, "logs")
        self.outputs_path = os.path.join(self.results_path_name, self.name, "outputs")
        self.data_tracking_path = os.path.join(self.results_path_name, self.name, "data_tracking")
        self.model_path = os.path.join(self.results_path_name, self.name, "neural_networks")
        self.cu_model_path = os.path.join(self.model_path, "cu_model")
        self.client_model_path = os.path.join(self.model_path, "client_model")
        self.model_indices_path = os.path.join(self.model_path, "model_indices.npy")
        Path(self.data_tracking_path).mkdir(parents=True, exist_ok=True)
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        fileh = logging.FileHandler(os.path.join(self.log_path, "log.txt"), 'a')
        global_logger.addHandler(fileh)
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

        self.metrics = {}

        self.models = [model_class().to(device) for _ in range(self.n_models)]
        self.model_weights = [{key: value for key, value in selected_model.named_parameters()}
                              for selected_model in self.models]
        if self.optimizer_name == "adam":
            self.optimizers = [optim.Adam(self.models[i].parameters(), lr=self.learning_rate, amsgrad=True) for i in
                               range(self.n_models)]
        elif self.optimizer_name == "sgd":
            self.optimizers = [optim.SGD(self.models[i].parameters(), lr=self.learning_rate) for i in
                               range(self.n_models)]
        else:
            raise NotImplementedError

        self.n_params_per_model = sum(param.numel() for param in self.models[0].parameters())
        self.n_params_compressed_gradient = self.n_params_per_model

        if self.use_reduced_G is True:
            if self.use_last_layer_only is True:
                n_params_compressed_gradient = 512000 + 1000
                self.random_G_indices = np.arange(self.n_params_per_model - n_params_compressed_gradient,
                                                  self.n_params_per_model)
            else:
                n_params_compressed_gradient = self.n_params_per_model // self.gradient_compression_ratio
                self.random_G_indices = np.random.choice(np.arange(self.n_params_per_model),
                                                         size=n_params_compressed_gradient, replace=False)
            self.n_params_compressed_gradient = n_params_compressed_gradient

        global_logger.info("n_params_per_model:{}".format(self.n_params_per_model))
        global_logger.info("n_params_compressed_gradient:{}".format(self.n_params_compressed_gradient))

        if self.load is True:
            # self.metrics = np.load(os.path.join(self.data_tracking_path, "metrics.pickle"))
            self.estimated_cluster_ids = np.load(self.model_indices_path, allow_pickle=True)
            for m_idx, model in enumerate(self.models):
                model.load_state_dict(torch.load(self.cu_model_path + "_" + str(m_idx) + ".pt"))
            for c_idx, client in enumerate(self.clients):
                client.local_model.load_state_dict(torch.load(self.client_model_path + "_" + str(c_idx) + ".pt"))
            global_logger.info("Load model success")
        else:
            self.estimated_cluster_ids = None
            global_logger.info("Fail to load model")

        self.writer = SummaryWriter(log_dir=self.log_path)
        self.writer.flush()

    def train(self):
        if self.algorithm == "CFLGP":
            self.run_CFLGP()
        elif self.algorithm == "IFCA":
            self.run_IFCA()
        elif self.algorithm == "MADMO":
            self.run_MADMO()
        elif self.algorithm == "FEDAVG":
            self.run_FEDAVG()
        else:
            raise NotImplementedError

    def test(self):
        # ====================================
        # Local Model Test
        # ====================================
        for client_idx, client in enumerate(tqdm(self.clients)):
            # _ = client.forward_propagation_model(device=self.device) # only for performance evaluation
            performance, info = client.output_analysis(dataset_type="test", device=self.device)
            output_path = os.path.join(self.outputs_path, "client_{}_info".format(client_idx))
            Path(output_path).mkdir(parents=True, exist_ok=True)
            np.save(output_path, info)
            global_logger.info("{}-th Client-performance:{}".format(client_idx, performance))

    def run_CFLGP(self):

        gradient_profile_matrix = np.zeros(shape=(self.n_models * self.n_params_compressed_gradient, self.n_clients))
        criterion_model_index = 0
        CLUSTERING_CONVERGENCE = False
        client_acc = None

        for c_round in range(self.n_communication_rounds):
            # ====================================
            # Local Model Test
            # ====================================
            if c_round % self.evaluation_interval == 0 or c_round == self.n_communication_rounds - 1:
                client_acc = np.zeros(shape=self.n_clients)
                for client_idx, client in enumerate(tqdm(self.clients)):
                    performance, _ = client.performance_evaluation(dataset_type="test", device=self.device)
                    client_acc[client_idx] = performance
                total_acc = np.sum(client_acc) / len(client_acc)
                global_logger.info(
                    "communication round: {}, Client-performances:{}, Avg:{}".format(c_round, client_acc, total_acc))
                # Model save
                for m_idx in range(self.n_models):
                    torch.save(self.models[m_idx].state_dict(), self.cu_model_path + "_" + str(m_idx) + ".pt")
                for c_idx, client_temp in enumerate(self.clients):
                    torch.save(client_temp.local_model.state_dict(), self.client_model_path + "_" + str(c_idx) + ".pt")
                np.save(self.model_indices_path, self.estimated_cluster_ids)

            # ====================================
            # Query
            # ====================================
            cr_start_time = time.time()
            for client_idx, client in enumerate(tqdm(self.clients)):
                # Select models for downlink transmission
                if c_round % self.clustering_period == 0 and CLUSTERING_CONVERGENCE is not True:
                    if c_round == 0:
                        query_model_indices = [0]
                    elif self.estimated_cluster_ids[client_idx] == criterion_model_index:
                        query_model_indices = [self.estimated_cluster_ids[client_idx]]
                    else:
                        query_model_indices = [criterion_model_index, self.estimated_cluster_ids[client_idx]]
                else:
                    query_model_indices = [self.estimated_cluster_ids[client_idx]]
                # local update or gradient calculation
                for model_idx in query_model_indices:
                    if c_round == 0:
                        estimated_cluster_id = 0
                    else:
                        estimated_cluster_id = self.estimated_cluster_ids[client_idx]
                    model_address = self.models[estimated_cluster_id]
                    weight_address = self.model_weights[estimated_cluster_id]

                    if self.protocol == "model_averaging":
                        model_info = client.compute_model_gap_by_local_update(model=model_address,
                                                                              weights=weight_address,
                                                                              device=self.device)
                    elif self.protocol == "gradient_averaging":
                        model_info = client.compute_gradient(model=model_address,
                                                             weights=weight_address,
                                                             device=self.device)
                    else:
                        raise NotImplementedError

                    if c_round % self.clustering_period == 0 and model_idx == criterion_model_index and CLUSTERING_CONVERGENCE is not True:
                        # global_logger.info("Graident Profile update.")
                        vectorized_model_info = flatten_tensor(model_info).clone().cpu().detach().numpy()

                        if self.use_reduced_G is True:
                            selected_vectorized_model_info = vectorized_model_info[self.random_G_indices]
                            # vectorized_model_info = np.zeros_like(vectorized_model_info)
                            # vectorized_model_info[:len(selected_vectorized_model_info)] = selected_vectorized_model_info
                            vectorized_model_info = selected_vectorized_model_info

                        # vectorized_model_info = vectorized_model_info / np.linalg.norm(vectorized_model_info)
                        # Cumulative Averaging
                        beta = (1 / (np.floor((c_round + 1) / (self.n_models * self.clustering_period)) + 1))
                        # global_logger.info("beta:{}".format(beta))
                        gradient_profile_matrix[(criterion_model_index) * self.n_params_compressed_gradient: (
                                                                                                                         criterion_model_index + 1) * self.n_params_compressed_gradient,
                        client_idx] = gradient_profile_matrix[
                                      (criterion_model_index) * self.n_params_compressed_gradient: (
                                                                                                           criterion_model_index + 1) * self.n_params_compressed_gradient,
                                      client_idx] * (1 - beta) \
                                      + (beta) * vectorized_model_info

            # ====================================
            # Clustering & Matching
            # ====================================
            if self.unknown_k is True and c_round < self.warmup_epoch:
                proposed_k = get_num_cluster(gradient_profile_matrix, n_centers=self.n_models,
                                             n_clients=self.n_clients,
                                             estimated_cluster_ids_old=self.estimated_cluster_ids)
                self.estimated_cluster_ids = np.zeros(shape=self.n_clients, dtype=int)
                self.estimated_cluster_ids_new = np.zeros(shape=self.n_clients, dtype=int)
                consistency_cnt = 0

            elif self.unknown_k is True and c_round == self.warmup_epoch:
                self.n_models = get_num_cluster(gradient_profile_matrix, n_centers=self.n_models,
                                                n_clients=self.n_clients,
                                                estimated_cluster_ids_old=self.estimated_cluster_ids)
                global_logger.info("Set number of clusters as {}.".format(self.n_models))
                for m in np.arange(1, self.n_models):
                    copy_weight(target=self.model_weights[m], source=self.model_weights[0])
                self.estimated_cluster_ids = np.zeros(shape=self.n_clients, dtype=int)
                self.estimated_cluster_ids_new = np.zeros(shape=self.n_clients, dtype=int)
                consistency_cnt = 0

            elif c_round % self.clustering_period == 0 and c_round < self.clustering_termination_threshold:
                # global_logger.info("Spectral Clustering.")
                info = spectral_clustering_and_matching(gradient_profile_matrix, n_centers=self.n_models,
                                                        n_clients=self.n_clients,
                                                        estimated_cluster_ids_old=self.estimated_cluster_ids)
                self.estimated_cluster_ids_new = info["estimated_cluster_ids"]  # update cluster ids.
                self.reduced_gradient_profile_matrix = info["reduced_gradient_profile_matrix"]
                self.singular_values = info["singular_values"]

                self.estimated_cluster_ids = self.estimated_cluster_ids_new  # np.zeros(shape=self.n_clients, dtype=int)

            global_logger.info("cluster ids: {}".format(self.estimated_cluster_ids))

            # ====================================
            # Model update
            # ====================================
            clustered_client_indices = [np.where(self.estimated_cluster_ids == cluster_id)[0] for cluster_id in
                                        range(self.n_models)]
            client_clusters = [[self.clients[i] for i in client_indices] for client_indices in clustered_client_indices]

            for cluster_idx, client_cluster in enumerate(client_clusters):
                if len(client_cluster) >= 1:
                    if self.protocol == "model_averaging":
                        model_averaging(self.model_weights[cluster_idx], client_cluster)
                    elif self.protocol == "gradient_averaging":
                        gradient_averaging(self.optimizers[cluster_idx], self.model_weights[cluster_idx],
                                           client_cluster, learning_rate=self.learning_rate)

            if self.unknown_k is True and c_round < self.warmup_epoch:
                pass
            else:
                # Encoder averaging
                if self.subprotocol == "encoder_averaging":
                    encoder_averaging(self.model_weights, self.clients)
                if self.subprotocol == "cnn_averaging":
                    cnn_averaging(self.model_weights, self.clients)
            # for client_idx, client in enumerate(tqdm(self.clients)):
            #     _ = client.forward_propagation_model(device=self.device) # only for performance evaluation

            time_for_communication_round = time.time() - cr_start_time
            print("{}th Communication round takes {} sec".format(c_round, time_for_communication_round))
            # ====================================
            # Data Tracking and save Results
            # ====================================
            data_tracking = {
                "estimated_cluster_ids": copy.deepcopy(self.estimated_cluster_ids),
                "client_accuracy": client_acc,
                "reduced_gradient_profile_matrix": copy.deepcopy(self.reduced_gradient_profile_matrix),
                "singular_values": copy.deepcopy(self.singular_values),
                "time_for_communication_round": time_for_communication_round
            }
            self.metrics["c_round_" + str(c_round)] = data_tracking
            with open(os.path.join(self.data_tracking_path, "metrics.pickle"), 'wb') as handle:
                pickle.dump(self.metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run_IFCA(self):
        for c_round in range(self.n_communication_rounds):
            # ====================================
            # Local Model Test
            # ====================================
            if c_round % self.evaluation_interval == 0 or c_round == self.n_communication_rounds - 1:
                client_acc = np.zeros(shape=self.n_clients)
                for client_idx, client in enumerate(tqdm(self.clients)):
                    # _ = client.forward_propagation_model(device=self.device) # only for performance evaluation
                    performance, info = client.performance_evaluation(dataset_type="test", device=self.device)
                    client_acc[client_idx] = performance
                total_acc = np.sum(client_acc) / len(client_acc)
                global_logger.info("communication round: {}, Client-performances:{}, Avg:{}".format(c_round, client_acc,
                                                                                                    total_acc))  # , np.sum(worst_client_acc)/len(worst_client_acc)) )
                # Model save
                # for m_idx in range(self.n_models):
                #     torch.save(self.models[m_idx].state_dict(), self.cu_model_path + "_" + str(m_idx) + ".pt")
                #     # torch.save(self.model[m_idx].state_dict(), self.full_model_path + str(m_idx) + "_epoch_" + str(c_round))
                # for c_idx, client_temp in enumerate(self.clients):
                #     torch.save(client_temp.local_model.state_dict(), self.client_model_path + "_" + str(c_idx) + ".pt")
                # np.save(self.model_indices_path, self.estimated_cluster_ids)

            # ============
            # Greedy Clustering
            # ============

            cr_start_time = time.time()
            accuracy_matrix = []
            self.estimated_cluster_ids = np.zeros(shape=self.n_clients, dtype=int)
            for client_idx, client in enumerate(tqdm(self.clients)):
                accuracies = np.zeros(shape=self.n_models)
                for model_idx in range(self.n_models):
                    copy_weight(target=client.local_model_weights, source=self.model_weights[model_idx])
                    _ = client.forward_propagation_model(device=self.device)  # only for performance evaluation
                    performance, info = client.performance_evaluation(dataset_type="train", device=self.device,
                                                                      single_batch=True)
                    accuracies[model_idx] = performance
                print("client idx:{}, acc{}".format(client_idx, accuracies))

                if self.task == "classification":
                    self.estimated_cluster_ids[client_idx] = np.argmax(accuracies)
                elif self.task == "compression" or self.task == "regression" or self.task == "vqvae_compression":
                    self.estimated_cluster_ids[client_idx] = np.argmin(accuracies)
                else:
                    raise NotImplementedError
                accuracy_matrix.append(accuracies)
            global_logger.info("cluster ids: {}".format(self.estimated_cluster_ids))
            # ============
            # Query
            # ============
            for client_idx, client in enumerate(tqdm(self.clients)):
                estimated_cluster_id = self.estimated_cluster_ids[client_idx]
                model_address = self.models[estimated_cluster_id]
                weight_address = self.model_weights[estimated_cluster_id]

                if self.protocol == "model_averaging":
                    model_info = client.compute_model_gap_by_local_update(model=model_address,
                                                                          weights=weight_address, device=self.device)
                elif self.protocol == "gradient_averaging":
                    model_info = client.compute_gradient(model=model_address,
                                                         weights=weight_address,
                                                         device=self.device)

            # ============
            # Model update
            # ============
            clustered_client_indices = [np.where(self.estimated_cluster_ids == cluster_id)[0] for cluster_id in
                                        range(self.n_models)]
            client_clusters = [[self.clients[i] for i in client_indices] for client_indices in clustered_client_indices]

            for cluster_idx, client_cluster in enumerate(client_clusters):
                if len(client_cluster) > 0:
                    if self.protocol == "model_averaging":
                        model_averaging(self.model_weights[cluster_idx], client_cluster)
                    elif self.protocol == "gradient_averaging":
                        gradient_averaging(self.optimizers[cluster_idx], self.model_weights[cluster_idx],
                                           client_cluster, learning_rate=self.learning_rate)

            # Encoder averaging
            if self.subprotocol == "encoder_averaging":
                encoder_averaging(self.model_weights, self.clients)
            if self.subprotocol == "cnn_averaging":
                cnn_averaging(self.model_weights, self.clients)
            for client_idx, client in enumerate(tqdm(self.clients)):
                _ = client.forward_propagation_model(device=self.device)  # only for performance evaluation

            time_for_communication_round = time.time() - cr_start_time
            print("{}th Communication round takes {} sec".format(c_round, time_for_communication_round))

            # ============
            # Data Tracking
            # ============
            data_tracking = {
                "estimated_cluster_ids": copy.deepcopy(self.estimated_cluster_ids),
                "client_accuracy": client_acc,
                "accuracy_matrix": accuracy_matrix,
                "time_for_communication_round": time_for_communication_round
            }
            self.metrics["c_round_" + str(c_round)] = data_tracking

            # ============
            # Save Results
            # ============

            with open(os.path.join(self.data_tracking_path, "metrics.pickle"), 'wb') as handle:
                pickle.dump(self.metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run_MADMO(self):
        EPS_1 = 0.4
        EPS_2 = 1.6

        self.estimated_cluster_ids = np.zeros(shape=self.n_clients, dtype=int)

        from learning_agent.utils import cluster_clients, compute_pairwise_similarities, \
            compute_max_update_norm, compute_mean_update_norm

        cluster_indices = [np.arange(self.n_clients).astype("int")]
        n_current_cluster = 1
        client_clusters = [[self.clients[i] for i in idcs] for idcs in cluster_indices]

        for c_round in range(self.n_communication_rounds):
            # ====================================
            # Local Model Test
            # ====================================
            if c_round % self.evaluation_interval == 0 or c_round == self.n_communication_rounds - 1:
                client_acc = np.zeros(shape=self.n_clients)
                for client_idx, client in enumerate(tqdm(self.clients)):
                    # _ = client.forward_propagation_model(device=self.device) # only for performance evaluation
                    performance, info = client.performance_evaluation(dataset_type="test", device=self.device)
                    client_acc[client_idx] = performance
                total_acc = np.sum(client_acc) / len(client_acc)
                global_logger.info("communication round: {}, Client-performances:{}, Avg:{}".format(c_round, client_acc,
                                                                                                    total_acc))  # , np.sum(worst_client_acc)/len(worst_client_acc)) )

            # ====================================
            # Query
            # ====================================
            cr_start_time = time.time()
            if c_round == 0:
                for client in self.clients:
                    client.synchronize_with_central_unit_model(model=self.models[0], weights=self.model_weights[0])

            for client_idx, client in enumerate(self.clients):
                estimated_cluster_id = self.estimated_cluster_ids[client_idx]
                if self.protocol == "model_averaging":
                    model_info = client.compute_model_gap_by_local_update(model=self.models[estimated_cluster_id],
                                                                          weights=self.model_weights[
                                                                              estimated_cluster_id], device=self.device)
                else:
                    model_info = client.compute_gradient(model=self.models[estimated_cluster_id],
                                                         weights=self.model_weights[estimated_cluster_id],
                                                         device=self.device)

            similarities = compute_pairwise_similarities(self.clients)
            cluster_indices_new = []

            max_norms = []
            mean_norms = []
            for idc in cluster_indices:
                max_norm = compute_max_update_norm([self.clients[i] for i in idc])
                max_norms.append(max_norm)
                mean_norm = compute_mean_update_norm([self.clients[i] for i in idc])
                mean_norms.append(mean_norm)
                global_logger.info("max_norm:{}, mean_norm:{}".format(max_norm, mean_norm))

                if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 5:
                    c1, c2 = cluster_clients(similarities[idc][:, idc])
                    cluster_indices_new += [idc[c1], idc[c2]]
                    # ========================
                    # Cluster index assignment
                    # ========================
                    original_model_idx = self.estimated_cluster_ids[idc[c1[0]]]
                    new_model_idx = n_current_cluster
                    for client_idx in idc[c2]:
                        self.estimated_cluster_ids[client_idx] = new_model_idx
                    n_current_cluster = n_current_cluster + 1
                    copy_weight(target=self.model_weights[new_model_idx], source=self.model_weights[original_model_idx])
                    ###
                else:
                    cluster_indices_new += [idc]

            global_logger.info("cluster ids: {}".format(self.estimated_cluster_ids))
            cluster_indices = cluster_indices_new

            for cnt, set_of_client_indices in enumerate(cluster_indices):
                estimated_cluster_id_for_a_cluster = self.estimated_cluster_ids[set_of_client_indices]
                estimated_cluster_id_for_a_cluster = set(estimated_cluster_id_for_a_cluster)
                assert len(estimated_cluster_id_for_a_cluster) == 1
                client_cluster = [self.clients[i] for i in set_of_client_indices]
                cluster_idx = estimated_cluster_id_for_a_cluster.pop()
                if self.protocol == "model_averaging":
                    model_averaging(self.model_weights[cluster_idx], client_cluster)

                elif self.protocol == "gradient_averaging":
                    gradient_averaging(self.optimizers[cluster_idx], self.model_weights[cluster_idx], client_cluster,
                                       learning_rate=self.learning_rate)

            # Encoder averaging
            if self.subprotocol == "encoder_averaging":
                encoder_averaging(self.model_weights, self.clients)
            if self.subprotocol == "cnn_averaging":
                cnn_averaging(self.model_weights, self.clients)
            for client_idx, client in enumerate(tqdm(self.clients)):
                _ = client.forward_propagation_model(device=self.device)  # only for performance evaluation

            time_for_communication_round = time.time() - cr_start_time
            print("{}th Communication round takes {} sec".format(c_round, time_for_communication_round))

            # ============
            # Data Tracking
            # ============
            data_tracking = {
                "estimated_cluster_ids": copy.deepcopy(self.estimated_cluster_ids),
                "client_accuracy": client_acc,
                "average_mean_norm": np.mean(mean_norms),
                "average_max_norm": np.mean(max_norms),
                "time_for_communication_round": time_for_communication_round
            }
            self.metrics["c_round_" + str(c_round)] = data_tracking

            # ============
            # Save Results
            # ============

            with open(os.path.join(self.data_tracking_path, "metrics.pickle"), 'wb') as handle:
                pickle.dump(self.metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run_FEDAVG(self):
        self.estimated_cluster_ids = np.zeros(shape=self.n_clients, dtype=int)
        for c_round in range(self.n_communication_rounds):
            if c_round % self.evaluation_interval == 0 or c_round == self.n_communication_rounds - 1:
                client_acc = np.zeros(shape=self.n_clients)
                for client_idx, client in enumerate(tqdm(self.clients)):
                    # _ = client.forward_propagation_model(device=self.device) # only for performance evaluation
                    performance, info = client.performance_evaluation(dataset_type="test", device=self.device)
                    client_acc[client_idx] = performance
                total_acc = np.sum(client_acc) / len(client_acc)
                global_logger.info("communication round: {}, Client-performances:{}, Avg:{}".format(c_round, client_acc,
                                                                                                    total_acc))  # , np.sum(worst_client_acc)/len(worst_client_acc)) )
                # Model save
                for m_idx in range(self.n_models):
                    torch.save(self.models[m_idx].state_dict(), self.cu_model_path + "_" + str(m_idx) + ".pt")
                    # torch.save(self.model[m_idx].state_dict(), self.full_model_path + str(m_idx) + "_epoch_" + str(c_round))
                for c_idx, client_temp in enumerate(self.clients):
                    torch.save(client_temp.local_model.state_dict(), self.client_model_path + "_" + str(c_idx) + ".pt")
                np.save(self.model_indices_path, self.estimated_cluster_ids)

            # ============
            # Query
            # ============
            cr_start_time = time.time()
            for client_idx, client in enumerate(self.clients):
                estimated_cluster_id = self.estimated_cluster_ids[client_idx]
                if self.protocol == "model_averaging":
                    model_info = client.compute_model_gap_by_local_update(model=self.models[estimated_cluster_id],
                                                                          weights=self.model_weights[
                                                                              estimated_cluster_id], device=self.device)
                elif self.protocol == "gradient_averaging":
                    model_info = client.compute_gradient(model=self.models[estimated_cluster_id],
                                                         weights=self.model_weights[estimated_cluster_id],
                                                         device=self.device)
                else:
                    raise NotImplementedError
            global_logger.info("cluster ids: {}".format(self.estimated_cluster_ids))

            if self.protocol == "model_averaging":
                model_averaging(self.model_weights[0], self.clients)
            elif self.protocol == "gradient_averaging":
                gradient_averaging(self.optimizers[0], self.model_weights[0], self.clients,
                                   learning_rate=self.learning_rate)
            else:
                raise NotImplementedError

            # Special case (Encoder averaging)
            if self.subprotocol == "encoder_averaging":
                encoder_averaging(self.model_weights, self.clients)
            for client_idx, client in enumerate(tqdm(self.clients)):
                _ = client.forward_propagation_model(device=self.device)  # only for performance evaluation

            # ============
            # Data Tracking
            # ============

            time_for_communication_round = time.time() - cr_start_time
            print("{}th Communication round takes {} sec".format(c_round, time_for_communication_round))
            data_tracking = {
                "estimated_cluster_ids": copy.deepcopy(self.estimated_cluster_ids),
                "client_accuracy": client_acc,
                "time_for_communication_round": time_for_communication_round
            }
            self.metrics["c_round_" + str(c_round)] = data_tracking

            # ============
            # Save Results
            # ============

            with open(os.path.join(self.data_tracking_path, "metrics.pickle"), 'wb') as handle:
                pickle.dump(self.metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
