# Clustered Federated Learning via Gradient-based Partitioning
This repository contains the implementation for the paper [Clustered Federated Learning via Gradient-based Partitioning, The 41st International Conference on Machine Learning (ICML 2024)](https://proceedings.mlr.press/v235/kim24p.html).


## TL;DR
### **Q**: Clustered Federated Learning (CFL) Problems: When training multiple models for diverse data distributions (clients), how do we cluster clients? (The Clients are connected to a central unit but *no local dataset is shared*).

### **A**: ① Accumulate client gradients over multiple learning iterations for a set of models. ② Apply spectral clustering to the accumulated gradient information.

This method quickly identifies the true client identity, leading to better task performance across various scenarios.


### Example results
Consider the MNIST dataset, which consists of handwritten digits across 10 classes. The data is distributed equally among eight groups of clients, with Rotation Transformations (RT) of 0, 15, 90, 105, 180, 195, 270, and 275 degrees applied to each group ($D$ = 8). This results in heterogeneous datasets due to the division and rotation.

The proposed CFL-GP algorithm effectively identifies client clusters using only *gradient information*, as illustrated below:


<img src="https://github.com/Heasung-Kim/clustered-federated-learning-via-gradient-based-partitioning/blob/main/imgs/figs/fig_12.png?raw=true" height="400" />

The visualization of reduced client features (i.e., the accumulated gradient information) obtained at different $t, t = 1, 5, 9$, and $13$ for two scenarios, where the number of clients is 64 and 128, respectively. Each data point represents a reduced client feature. **Notably, CFL-GP achieves optimal clustering (ARI of 1.0) within a single clustering round $t = 1$ using these features.**

This capability of swiftly identifying client clusters results in faster convergence and improved task performance across various benchmarks.



## Quick Start
To get started, simply `run main.py` with the appropriate parameters. You can customize the experimental environment using command-line arguments.

For a comprehensive example, refer to `main_notebook.ipynb`. This Jupyter notebook demonstrates experiments on the rotated MNIST dataset with eight different distributions, four models, 128 clients, and a batch size of 100, which is the default setup in `main.py`. The notebook runs CFL-GP and baseline methods for performance comparison, visualizes gradient profile matrices, and tracks performance over communication rounds.







## Citing Our Work
If you find this repository helpful, please cite our work:

    @inproceedings{kim2024clustered,
    title={Clustered Federated Learning via Gradient-based Partitioning},
    author={Kim, Heasung and Kim, Hyeji and De Veciana, Gustavo},
    booktitle={Forty-first International Conference on Machine Learning}}


## References

This repository was inspired by the following papers and codebases:

Sattler, Felix, Klaus-Robert Müller, and Wojciech Samek. "Clustered federated learning: Model-agnostic distributed multitask optimization under privacy constraints." IEEE transactions on neural networks and learning systems 32.8 (2020): 3710-3722. https://github.com/felisat/clustered-federated-learning

Ghosh, Avishek, et al. "An efficient framework for clustered federated learning." Advances in Neural Information Processing Systems 33 (2020): 19586-19597, https://github.com/jichan3751/ifca

