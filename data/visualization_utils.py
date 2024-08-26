import os
from global_config import ROOT_DIRECTORY
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

projects_directory = os.path.abspath(os.path.join(ROOT_DIRECTORY, os.pardir))
results_save_path = os.path.join(projects_directory, "results")
fig_directory = os.path.join(results_save_path, "data", "best_worst_samples")
if not os.path.isdir(fig_directory):
    os.makedirs(fig_directory)


def get_singular_values(result_path):
    with open(result_path, 'rb') as handle:
        result = pickle.load(handle)

    gradient_profile_matrices = []
    for c_round_data in result.items():
        gradient_profile_matrices.append(c_round_data[1]["singular_values"])

    return gradient_profile_matrices


def get_gradient_profile_matrix(result_path):
    with open(result_path, 'rb') as handle:
        result = pickle.load(handle)

    gradient_profile_matrices = []
    for c_round_data in result.items():
        gradient_profile_matrices.append(c_round_data[1]["reduced_gradient_profile_matrix"])

    return gradient_profile_matrices


def get_cluster_id_matrix(result_path):
    with open(result_path, 'rb') as handle:
        result = pickle.load(handle)

    changes_of_estimated_cluster_ids = []
    for c_round_data in result.items():
        changes_of_estimated_cluster_ids.append(c_round_data[1]["estimated_cluster_ids"])
    changes_of_estimated_cluster_ids = np.squeeze(changes_of_estimated_cluster_ids)
    changes_of_estimated_cluster_ids = changes_of_estimated_cluster_ids.T
    return changes_of_estimated_cluster_ids


def get_average_performance_vector(result_path):
    with open(result_path, 'rb') as handle:
        result = pickle.load(handle)

    changes_of_estimated_cluster_ids = []
    for c_round_data in result.items():
        acc = c_round_data[1]["client_accuracy"]
        changes_of_estimated_cluster_ids.append(np.mean(acc))
    changes_of_estimated_cluster_ids = np.squeeze(changes_of_estimated_cluster_ids)
    return changes_of_estimated_cluster_ids


def get_performance_vector(result_path):
    with open(result_path, 'rb') as handle:
        result = pickle.load(handle)

    changes_of_estimated_cluster_ids = []
    for c_round_data in result.items():
        acc = c_round_data[1]["client_accuracy"]
        changes_of_estimated_cluster_ids.append(acc)
    # changes_of_estimated_cluster_ids = np.squeeze(changes_of_estimated_cluster_ids)
    return np.array(changes_of_estimated_cluster_ids)


def visualize_mnist(dataset):
    # Randomly select 9 images
    # dataset = np.squeeze(dataset)
    dataset = np.reshape(dataset, newshape=(dataset.shape[0], 28, 28))
    n_plots = np.minimum(dataset.shape[0], 9)
    indices = np.random.choice(dataset.shape[0], np.minimum(dataset.shape[0], 9), replace=False)
    selected_images = dataset[indices]

    # Create a 3x3 subplot
    if n_plots == 9:
        fig, axes = plt.subplots(3, 3, figsize=(6, 6))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(selected_images[i], cmap='gray', interpolation='none')
            ax.axis('off')
    else:
        selected_images = selected_images[0]
        fig, axes = plt.subplots(1, 1)
        axes.imshow(selected_images, cmap='gray', interpolation='none')
        axes.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import os
    from global_config import ROOT_DIRECTORY

    randidx = np.random.choice(a=np.arange(1000), size=10)
    recovered_imgs = np.load(os.path.join(ROOT_DIRECTORY, "data", "recovered_imgs_c1.npy"))
    original_imgs = np.load(os.path.join(ROOT_DIRECTORY, "data", "original_imgs_c1.npy"))

    label_pred(label=original_imgs[randidx], pred=recovered_imgs[randidx], plot_name="temp")
