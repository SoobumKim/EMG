import os
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

import wandb
from data_preprocess import preprocess_dataset
from src.model.models import EMGCombinedBiLSTMModel, EMGCombinedModel, EMGLSTMModel
from src.model.training_utils import setup_training

with open("src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_dir = config["data"]["root_dir"]
csv_dirs = glob(os.path.join(data_dir, config["data"]["emg_dir"], "*_M_*.csv"))
meta_data = pd.read_csv(os.path.join(data_dir, config["data"]["metadata_dir"]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model_config = config["model"]
model = eval(model_config["name"])(
    input_size=model_config["input_size"],
    hidden_size=model_config["hidden_size"],
    num_layers=model_config["num_layers"],
    dropout_rate=model_config["dropout"],
).to(device)

train_config = config["train"]

# Data preprocessing
dataset_builder = preprocess_dataset(csv_dirs, meta_data)
dataset_builder.read_csv()
dataset_builder.split_dataset()
loader = dataset_builder.load_dataset(train_config["batch_size"])

train_loader, valid_loader, test_loader = loader

criterion, optimizer, scheduler = setup_training(train_config, model)

nowtime = datetime.now().strftime("%m%d")
# wandb init
wandb.init(
    project="emg-age-estimation",
    name="{}_{}_{}_{}_{}_{}".format(
        model_config["name"],
        model_config["hidden_size"],
        model_config["num_layers"],
        train_config["criterion"],
        train_config["optimizer"],
        nowtime,
    ),
    config=config,
)


# train
pre_valid_mae = 1e9
for epoch in tqdm(range(train_config["epochs"])):
    model.train()
    total_loss = 0
    total_len = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)  # (batch_size, seq_len)
        y_batch = y_batch.to(device).unsqueeze(1)  # (batch_size, 1)

        optimizer.zero_grad()
        output = model(X_batch[:, :2048])  # (batch_size, 1)
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

        X_batch = X_batch.to(device)  # (batch_size, seq_len)
        y_batch = y_batch.to(device).unsqueeze(1)  # (batch_size, 1)

        output = model(X_batch[:, :2048])  # (batch_size, 1)
        val_loss = criterion(output, y_batch)

        total_val_loss += val_loss.item()
        total_val_len += len(valid_loader)

        predictions.extend(output.detach().cpu().numpy().flatten())
        targets.extend(y_batch.detach().cpu().numpy().flatten())

    valid_mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
    valid_loss = total_val_loss / total_val_len

    scheduler.step(valid_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_mae": valid_mae,
            "learning_rate": current_lr,
        }
    )

    print(
        f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f} Valid MAE: {valid_mae:.4f}"
    )

    if valid_mae < pre_valid_mae:
        torch.save(
            model.state_dict(),
            os.path.join(
                config["output"]["path"],
                "{}_{}_{}_{}_{}_{}.pt".format(
                    model_config["name"],
                    model_config["hidden_size"],
                    model_config["num_layers"],
                    train_config["criterion"],
                    train_config["optimizer"],
                    nowtime,
                ),
            ),
        )
        pre_valid_mae = valid_mae

# test
model.eval()
predictions, targets = [], []

print("Test...")
with torch.no_grad():
    for X_test, y_test in tqdm(test_loader):
        X_test = X_test.to(device)
        y_test = y_test.to(device).unsqueeze(1)

        pred = model(X_test[:, :2048])
        predictions.extend(pred.detach().cpu().numpy().flatten())
        targets.extend(y_test.detach().cpu().numpy().flatten())

# 예: MAE 계산
mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))

wandb.log({"test_mae": mae})
print(f"Test MAE: {mae:.2f}")

wandb.save(
    name="{}_{}_{}_{}_{}_{}".format(
        model_config["name"],
        model_config["hidden_size"],
        model_config["num_layers"],
        train_config["criterion"],
        train_config["optimizer"],
        nowtime,
    )
)
