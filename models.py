import torch
import torch.nn as nn

class EMGLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 나이 회귀 예측

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = x.unsqueeze(-1)  # → (batch_size, seq_len, input_size=1)
        lstm_out, _ = self.lstm(x)  # → (batch_size, seq_len, hidden_size)
        last_output = lstm_out[:, -1, :]  # 마지막 timestep 출력만 사용
        out = self.fc(last_output)  # → (batch_size, 1)
        return out