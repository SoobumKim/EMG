import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


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
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError("Unsupported optimizer")

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],  # 또는 total_epochs
        eta_min=1e-6,  # 최소 학습률
    )
    return criterion, optimizer, scheduler
