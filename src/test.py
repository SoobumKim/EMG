import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from data_preprocess import preprocess_dataset
from src.model.models import EMGCombinedModel, EMGLSTMModel

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

_, _, test_loader = loader

model.load_state_dict(torch.load("model_adam.pt"))
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
