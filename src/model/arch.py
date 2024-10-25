# models/cnn_lstm.py
import torch
import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # LSTM for temporal learning
        self.lstm = nn.LSTM(input_size=64*64, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 1)  # Output a single prediction for each timestep

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        cnn_out = []
        for t in range(seq_len):
            cnn_features = self.cnn(x[:, t, :, :, :])  # Apply CNN to each timestep
            cnn_features = cnn_features.view(batch_size, -1)
            cnn_out.append(cnn_features)
        
        cnn_out = torch.stack(cnn_out, dim=1)  # Stack CNN outputs over timesteps
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = lstm_out[:, -1, :]  # Take the last output of LSTM
        out = self.fc(lstm_out)
        return out
