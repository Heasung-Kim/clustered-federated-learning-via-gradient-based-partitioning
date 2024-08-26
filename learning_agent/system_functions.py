import numpy as np
import math
import scipy.optimize as sopt
import torch
import torch.nn
import torch.nn.functional as F
import random


def get_vectorized_gradient(gradient_01, g1_cache):
    pointer = 0
    for i in range(len(gradient_01)):
        g1 = gradient_01[i]
        if g1 is not None:
            g1 = g1.flatten()
            g1_cache[pointer:pointer + len(g1)] = g1
            pointer += len(g1)
    g1_cache = g1_cache / np.linalg.norm(g1_cache, ord=2)
    return g1_cache


def result_analysis_vqvae_compression(model, data_loader, device, single_batch=False):
    model.eval()
    cnt = 0
    original_imgs = []
    recovered_imgs = []
    codewords = []
    representations = []
    H_SF = []

    with torch.no_grad():
        for i, (x_HSF, y) in enumerate(data_loader):
            x, H_SF_sample = x_HSF
            x, y, H_SF_sample = x.to(device), y.to(device), H_SF_sample.to(device)

            embedding_loss, x_hat, perplexity, codeword, representation, z_q = model(x, VARIOUS_OUTPUT=True)

            original_imgs.append(x)
            recovered_imgs.append(x_hat)
            codewords.append(codeword)
            representations.append(representation)
            H_SF.append(H_SF_sample)

            if single_batch is True:
                break

            cnt += 1
    original_imgs = torch.concat(original_imgs, 0)
    recovered_imgs = torch.concat(recovered_imgs, 0)
    codewords = torch.concat(codewords, 0)
    representations = torch.concat(representations, 0)
    H_SF = torch.concat(H_SF, 0)

    # centering for the COST2100 channel dataset
    original_imgs = original_imgs - 0.5
    recovered_imgs = recovered_imgs - 0.5

    # Calculate the NMSE
    NMSE = get_NMSE(original_imgs, recovered_imgs)
    recovered_imgs = recovered_imgs.cpu().detach().numpy()
    recovered_imgs = np.transpose(recovered_imgs, (0, 2, 3, 1))
    recovered_imgs = recovered_imgs[:, :, :, 0] + 1j * recovered_imgs[:, :, :, 1]

    original_imgs = original_imgs.cpu().detach().numpy()
    original_imgs = np.transpose(original_imgs, (0, 2, 3, 1))
    original_imgs = original_imgs[:, :, :, 0] + 1j * original_imgs[:, :, :, 1]

    complex_ad_channel = torch.from_numpy(recovered_imgs)

    zero_padding = torch.zeros(len(complex_ad_channel), 32, 256 - 32)
    input_tensor = torch.cat((complex_ad_channel, zero_padding), dim=2)
    # Perform 2D FFT along the last two dimensions (axes=(1, 2))

    # H_SF_hat = torch.fft.fft2(input_tensor, dim=(-1, -2))

    H_SF_hat = np.fft.fft2(input_tensor.cpu().detach().numpy())

    # H_SF = data_loader.dataset.dataset.imgs_spatial_frequency
    H_SF = H_SF.cpu().detach().numpy()
    H_SF = H_SF[:, 0] + 1j * H_SF[:, 1]

    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    for sample_idx in range(1):
        # Create subplots to display both tensors
        fig, axs = plt.subplots(4, 1, figsize=(10, 10))

        # Plot the second tensor as a heatmap
        axs[0].imshow(abs(H_SF[sample_idx]), cmap='viridis')
        axs[0].set_title('Original')

        # Plot the second tensor as a heatmap
        # axs[3].imshow(abs(recovered_imgs[sample_idx]), cmap='viridis')
        axs[1].imshow(abs(np.fft.ifft2(H_SF[sample_idx])), cmap='viridis')
        axs[1].set_title('Original AD')

        # Plot the second tensor as a heatmap
        axs[2].imshow(abs(original_imgs[sample_idx]), cmap='viridis')
        axs[2].set_title('Recovered AD')

        # Plot the first tensor as a heatmap
        axs[3].imshow(abs(H_SF_hat[sample_idx]), cmap='viridis')
        axs[3].set_title('Recovered')

        # Display the plots
        plt.savefig('sample{}.pdf'.format(), bbox_inches='tight')
        plt.show()

    info = {
        "NMSE": NMSE,
        "original_imgs": original_imgs,
        "recovered_imgs": recovered_imgs,
        "codewords": codewords,
        "representations": representations,
    }
    model.train()
    return NMSE, info


def result_analysis(model, data_loader, task, device, single_batch=False):
    if task == "classification":
        raise NotImplementedError
    elif task == "compression" or task == "DORO_compression":
        return NotImplementedError
    elif task == "vqvae_compression" or task == "DORO_vqvae_compression":
        return result_analysis_vqvae_compression(model=model, data_loader=data_loader, device=device,
                                                 single_batch=single_batch)
    elif task == "regression":
        raise NotImplementedError


def copy_weight(target, source, namespace=None, exc_namespace=None):
    if namespace is not None:
        for name in target:
            if namespace in name:
                target[name].data = source[name].data.clone()

    elif exc_namespace is not None:
        for name in target:
            if exc_namespace not in name:
                target[name].data = source[name].data.clone()
    else:
        for name in target:
            target[name].data = source[name].data.clone()


def copy_value(target, source):
    for cnt, name in enumerate(target):
        target[name].data = source[cnt].data.clone()


def get_model_difference(model_size_storage, updated_model_weights, original_model_weights):
    for name in model_size_storage:
        model_size_storage[name].data = updated_model_weights[name].data.clone() - original_model_weights[
            name].data.clone()
    return model_size_storage


def train_model(model, data_loader, optimizer, scheduler, epochs, task, device, single_batch=False,
                train_data_variance=1):
    # Reference
    # [1] Zhai, Runtian, et al. "Doro: Distributional and outlier robust optimization." ICML 2021
    #       https://github.com/RuntianZ/doro/blob/master/dro.py
    #
    # Distributional and outlier robust optimization parameters
    eps, alpha, max_l = 0.0, 0.05, 10.0
    C = math.sqrt(1 + (1 / alpha - 1) ** 2)

    model.train()
    for ep in range(epochs):
        running_loss, samples = 0.0, 0

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            if task == "classification":
                loss = torch.nn.CrossEntropyLoss()(model(x), y)
            elif task == "compression":
                loss = torch.nn.MSELoss()(model(x), x)
            elif task == "DORO_compression":
                loss = torch.nn.MSELoss(reduction='none')(model(x), x)
                loss = torch.mean(loss, dim=(1, 2, 3))
            elif task == "regression":
                loss = torch.nn.MSELoss()(model(x), y)
            elif task == "vqvae_compression":
                embedding_loss, x_hat, perplexity = model(x)
                mse_loss = torch.nn.MSELoss()(x_hat, x)
                loss = mse_loss + embedding_loss * (train_data_variance)
            else:
                raise NotImplementedError

            if "DORO" in task:
                batch_size = len(x)
                n = int(eps * batch_size)
                rk = torch.argsort(loss, descending=True)
                l0 = loss[rk[n:]]
                foo = lambda eta: C * math.sqrt((F.relu(l0 - eta) ** 2).mean().item()) + eta
                opt_eta = sopt.brent(foo, brack=(0, max_l))
                loss = C * torch.sqrt((F.relu(l0 - opt_eta) ** 2).mean()) + opt_eta

            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if single_batch is True:
                return running_loss / samples
    # for param_group in optimizer.param_groups:
    #    print("Learning rate:", param_group['lr'])
    return running_loss / samples


def forward_model(model, data_loader, optimizer, epochs, task, device, single_batch=False):
    model.train()
    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            if task == "classification":
                loss = torch.nn.CrossEntropyLoss()(model(x), y)
            elif task == "compression" or task == "DORO_compression":
                loss = torch.nn.MSELoss()(model(x), x)
            elif task == "regression":
                loss = torch.nn.MSELoss()(model(x), y)
            elif task == "vqvae_compression":
                embedding_loss, x_hat, perplexity = model(x)
                loss = torch.nn.MSELoss()(x_hat, x)
            else:
                raise NotImplementedError
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]

            if single_batch is True:
                return running_loss / samples

    return running_loss / samples


def test_model_classification(model, data_loader, device, single_batch=False):
    model.eval()
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)

            y_ = model(x)
            _, predicted = torch.max(y_.data, 1)

            samples += y.shape[0]
            correct += (predicted == y).sum().item()

            if single_batch is True:
                break
    model.train()
    info = None

    return correct / samples, info


def test_model_vqvae_compression(model, data_loader, device, single_batch=False):
    model.eval()
    cnt = 0
    original_imgs = []
    recovered_imgs = []
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            embedding_loss, x_hat, perplexity = model(x)
            original_imgs.append(x)
            recovered_imgs.append(x_hat)

            if single_batch is True:
                break

            cnt += 1
    original_imgs = torch.concat(original_imgs, 0)
    recovered_imgs = torch.concat(recovered_imgs, 0)

    # centering for the COST2100 channel dataset
    original_imgs = original_imgs - 0.5
    recovered_imgs = recovered_imgs - 0.5

    # Calculate the NMSE
    NMSE = get_NMSE(original_imgs, recovered_imgs)
    info = None

    model.train()
    return NMSE, info


def get_NMSE(original_imgs, recovered_imgs):
    try:
        power_gt = original_imgs[:, 0, :, :] ** 2 + original_imgs[:, 1, :, :] ** 2
        difference = original_imgs - recovered_imgs
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
    except:
        power_gt = original_imgs[:, 0, :, :] ** 2
        difference = original_imgs - recovered_imgs
        mse = difference[:, 0, :, :] ** 2

    nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())
    NMSE = nmse.cpu().detach().numpy()
    return NMSE


def test_model_compression(model, data_loader, device, single_batch=False):
    model.eval()
    cnt = 0
    original_imgs = []
    recovered_imgs = []
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            x_hat = model(x)
            original_imgs.append(x)
            recovered_imgs.append(x_hat)
            if single_batch is True:
                break
            cnt += 1
    original_imgs = torch.concat(original_imgs, 0)
    recovered_imgs = torch.concat(recovered_imgs, 0)

    # centering for the COST2100 channel dataset
    original_imgs = original_imgs - 0.5
    recovered_imgs = recovered_imgs - 0.5

    # Calculate the NMSE
    power_gt = original_imgs[:, 0, :, :] ** 2 + original_imgs[:, 1, :, :] ** 2
    difference = original_imgs - recovered_imgs
    mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
    nmse = (mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2]))

    nmse = nmse.mean()
    nmse = 10 * torch.log10(nmse)

    # worst_users_nmse = worst_users_nmse.cpu().detach().numpy()
    nmse = nmse.cpu().detach().numpy()
    # info = {"worst_10%_sample_nmse" : worst_users_nmse}
    info = None
    model.train()
    return nmse, info


def test_model_regression(model, data_loader, device, single_batch=False):
    model.eval()
    cnt = 0
    targets = []
    preds = []
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            y_hat = model(x)
            targets.append(y)
            preds.append(y_hat)

            if single_batch is True:
                break

            cnt += 1
    targets = torch.concat(targets, 0)
    preds = torch.concat(preds, 0)
    preds = torch.reshape(preds, targets.shape)

    mse = torch.mean((targets - preds) ** 2)
    mse = mse.cpu().detach().numpy()

    model.train()
    info = None
    return mse, info


def test_model(model, data_loader, task, device, single_batch=False):
    if task == "classification":
        return test_model_classification(model=model, data_loader=data_loader, device=device, single_batch=single_batch)
    elif task == "compression" or task == "DORO_compression":
        return test_model_compression(model=model, data_loader=data_loader, device=device, single_batch=single_batch)
    elif task == "vqvae_compression":
        return test_model_vqvae_compression(model=model, data_loader=data_loader, device=device,
                                            single_batch=single_batch)
    elif task == "regression":
        return test_model_regression(model=model, data_loader=data_loader, device=device, single_batch=single_batch)


def cnn_averaging(central_unit_model_weights, client_cluster):
    client_local_model_weights_list = [client.local_model_weights for client in client_cluster]
    for model_weight in central_unit_model_weights:
        for name in model_weight:
            if ('fc' in name) is not True:
                average_encoder_weight = torch.mean(
                    torch.stack([client_local_model_weights[name].data for client_local_model_weights in
                                 client_local_model_weights_list]),
                    dim=0).clone()
                model_weight[name].data = average_encoder_weight

    for client in client_cluster:
        copy_weight(target=client.local_model_weights, source=central_unit_model_weights[0], exc_namespace="fc")


def encoder_averaging(central_unit_model_weights, client_cluster):
    client_local_model_weights_list = [client.local_model_weights for client in client_cluster]
    for model_weight in central_unit_model_weights:
        for name in model_weight:
            if 'encoder' in name:
                average_encoder_weight = torch.mean(
                    torch.stack([client_local_model_weights[name].data for client_local_model_weights in
                                 client_local_model_weights_list]),
                    dim=0).clone()
                model_weight[name].data = average_encoder_weight
            if 'embedding' in name:
                average_encoder_weight = torch.mean(
                    torch.stack([client_local_model_weights[name].data for client_local_model_weights in
                                 client_local_model_weights_list]),
                    dim=0).clone()
                model_weight[name].data = average_encoder_weight

    for client in client_cluster:
        copy_weight(target=client.local_model_weights, source=central_unit_model_weights[0], namespace="encoder")


def model_averaging(model_weight_address, client_cluster):
    client_model_weight_address_list = [client.model_size_storage for client in client_cluster]
    for name in model_weight_address:
        tmp = torch.mean(
            torch.stack([model_difference[name].data for model_difference in client_model_weight_address_list]),
            dim=0).clone()
        model_weight_address[name].data += tmp

    for client in client_cluster:
        copy_weight(target=client.local_model_weights, source=model_weight_address)


def prox_model_averaging(model_weight_address, client_cluster, selection_ratio=0.5):
    num_items_to_select = int(len(client_cluster) * selection_ratio)
    # Select items randomly from the list
    client_cluster = random.sample(client_cluster, num_items_to_select)
    client_model_weight_address_list = [client.model_size_storage for client in client_cluster]
    for name in model_weight_address:
        tmp = torch.mean(
            torch.stack([model_difference[name].data for model_difference in client_model_weight_address_list]),
            dim=0).clone()
        model_weight_address[name].data += tmp

    for client in client_cluster:
        copy_weight(target=client.local_model_weights, source=model_weight_address)


def gradient_averaging(optimizer, model_weight_address, client_cluster, learning_rate):
    gradients = [client.model_size_storage for client in client_cluster]
    for cnt, name in enumerate(model_weight_address):
        tmp = torch.mean(torch.stack([gradient[name].data for gradient in gradients]), dim=0).clone()
        model_weight_address[name].grad = tmp
    optimizer.step()
    for client in client_cluster:
        copy_weight(target=client.local_model_weights, source=model_weight_address)


def flatten_tensor(source):
    if type(source) is tuple:
        ft = []
        for value in source:
            ft.append(value.flatten())
        ft = torch.cat(ft)
    else:
        ft = torch.cat([value.flatten() for value in source.values()])
    return ft
