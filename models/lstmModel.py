from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.5):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=4, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)  # 2 classes for CrossEntropyLoss

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = self.dropout(last_output)
        output = self.fc(x)
        return output