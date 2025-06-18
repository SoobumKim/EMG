import os
from glob import glob
import pandas as pd
import yaml
from tqdm import tqdm 
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data_preprocess import preprocess_dataset
from models import EMGLSTMModel

data_root_dir = "EMG"
csv_dirs = glob(os.path.join(data_root_dir,"*_M_*.csv"))
meta_data = pd.read_csv("metadata.csv")

# Data preprocessing
dataset_builder = preprocess_dataset(csv_dirs, meta_data)
dataset_builder.read_csv()
dataset_builder.split_dataset()
loader = dataset_builder.load_dataset()

train_loader, valid_loader, test_loader = loader

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model_config = config["model"]
model = EMGLSTMModel(
            input_size=model_config["input_size"], 
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"]
        ).to(device)


train_config = config["train"]

# Loss 함수 설정
if config["train"]["criterion"] == "mse":
    criterion = nn.MSELoss()
elif config["train"]["criterion"] == "mae":
    criterion = nn.L1Loss()
else:
    raise ValueError("Unsupported loss function")

# Optimizer 설정
if config["train"]["optimizer"] == "adam":
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
elif config["train"]["optimizer"] == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=config["train"]["learning_rate"])
else:
    raise ValueError("Unsupported optimizer")

# wandb init
wandb.init(
    project="emg-age-estimation",
    name="TODO_naming", 
    config=config
)

pre_valid_loss = 1e9

# train
for epoch in tqdm(range(train_config["epochs"])):
    model.train()
    total_loss = 0
    total_len = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)         # (batch_size, seq_len)
        y_batch = y_batch.to(device).unsqueeze(1)  # (batch_size, 1)

        optimizer.zero_grad()
        output = model(X_batch)              # (batch_size, 1)
        loss = criterion(output, y_batch.squeeze(1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_len += len(train_loader)

    train_loss = total_loss / total_len

    predictions, targets = [], []
    model.eval()
    for X_batch, y_batch in valid_loader:
        total_val_loss = 0
        total_val_len = 0

        X_batch = X_batch.to(device)         # (batch_size, seq_len)
        y_batch = y_batch.to(device).unsqueeze(1)  # (batch_size, 1)

        output = model(X_batch)              # (batch_size, 1)
        val_loss = criterion(output, y_batch)

        total_val_loss += val_loss.item()
        total_val_len += len(valid_loader)

        predictions.extend(output.detach().cpu().numpy().flatten())
        targets.extend(y_batch.detach().cpu().numpy().flatten())
    
    valid_mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
    valid_loss = total_val_loss / total_val_len

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "valid_mae": valid_mae,
    })

    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f} Valid MAE: {valid_mae:.4f}")

    if valid_loss < pre_valid_loss:
        torch.save(model.state_dict(), "model.pt")
    pre_valid_loss = valid_loss

# test
model.eval()
predictions, targets = [], []

print("Test...")
with torch.no_grad():
    for X_test, y_test in tqdm(test_loader):
        X_test = X_test.to(device)
        y_test = y_test.to(device).unsqueeze(1)

        pred = model(X_test)
        predictions.extend(pred.detach().cpu().numpy().flatten())
        targets.extend(y_test.detach().cpu().numpy().flatten())

# 예: MAE 계산
mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))

wandb.log({"test_mae": mae})
print(f"Test MAE: {mae:.2f}")


wandb.save("model.pt")