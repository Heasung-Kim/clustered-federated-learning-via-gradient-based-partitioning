
def get_model_class(config):
    model_class = None
    if config["model"] == "SimpleLinear":
        from model.neural_networks.simple_neural_classification.simple_linear import SimpleLinear
        model_class = SimpleLinear

    if config["model"] == "LinearRegressor":
        from model.neural_networks.linear_regressor.linear_regressor import LinearRegressor
        model_class = LinearRegressor

    elif config["model"] == "MADMOConvNet":
        from model.neural_networks.madmo_convnet.madmo_convnet import MADMOConvNet
        model_class = MADMOConvNet

    elif config["model"] == "Resnet18":
        from model.neural_networks.resnet18.resnet18 import Resnet18
        model_class = Resnet18

    elif config["model"] == "EfficientNet":
        from model.neural_networks.efficientnet.efficientnet import EfficientNet
        model_class = EfficientNet

    return model_class
