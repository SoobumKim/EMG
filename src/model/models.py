import torch
import torch.nn as nn


class EMGLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout_rate=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)  # 나이 회귀 예측
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = x.unsqueeze(-1).contiguous()  # → (batch_size, seq_len, input_size=1)
        lstm_out, _ = self.lstm(x)  # → (batch_size, seq_len, hidden_size)
        last_output = lstm_out[:, -1, :]  # 마지막 timestep 출력만 사용
        x = self.dropout(last_output)
        out = self.fc(last_output)  # → (batch_size, 1)
        return out


class EMGCombinedModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout_rate=0.5,
        feature_size=2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

        # 통계 feature 포함된 최종 FC
        self.fc1 = nn.Linear(hidden_size + feature_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def compute_features(self, emg_batch):  # shape: (B, T)
        feats = []
        for emg in emg_batch:
            rms = torch.sqrt(torch.mean(emg**2))
            mav = torch.mean(torch.abs(emg))
            # zc = torch.sum(torch.diff(torch.sign(emg)) != 0).float()
            # wl = torch.sum(torch.abs(torch.diff(emg))).float()
            feats.append([rms, mav])
        return torch.tensor(feats, dtype=torch.float32, device=emg_batch.device)

    def forward(self, x_seq):
        # x_seq: (B, T) → (B, T, 1)
        x = x_seq.unsqueeze(-1).contiguous()
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (B, hidden_size)

        # x_feats: (B, feature_size) → 그대로 사용
        x = torch.cat([last_hidden, self.compute_features(x_seq)], dim=1)  # concat

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class EMGCombinedBiLSTMModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout_rate=0.5,
        feature_size=2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=True,  # ✅ BiLSTM 활성화
            batch_first=True,
        )

        # BiLSTM이므로 hidden_size × 2
        self.fc1 = nn.Linear(hidden_size * 2 + feature_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def compute_features(self, emg_batch):  # shape: (B, T)
        feats = []
        for emg in emg_batch:
            rms = torch.sqrt(torch.mean(emg**2))
            mav = torch.mean(torch.abs(emg))
            feats.append([rms, mav])
        return torch.tensor(feats, dtype=torch.float32, device=emg_batch.device)

    def forward(self, x_seq):
        x = x_seq.unsqueeze(-1).contiguous()  # (B, T, 1)
        lstm_out, _ = self.lstm(x)

        # BiLSTM: 마지막 타임스텝의 양방향 hidden state를 연결
        last_hidden = lstm_out[:, -1, :]  # shape: (B, hidden_size * 2)

        x_feats = self.compute_features(x_seq)  # shape: (B, feature_size)
        x = torch.cat(
            [last_hidden, x_feats], dim=1
        )  # shape: (B, hidden_size*2 + feature_size)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
