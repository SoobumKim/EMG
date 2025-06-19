import torch.nn as nn
import torch.optim as optim


def setup_training(config, model):
    # Loss
    if config["criterion"] == "mse":
        criterion = nn.MSELoss()
    elif config["criterion"] == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError("Unsupported loss function")

    # Optimizer
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["train"]["learning_rate"])
    else:
        raise ValueError("Unsupported optimizer")

    return criterion, optimizer
