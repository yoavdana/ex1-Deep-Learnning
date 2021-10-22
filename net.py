import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NetWork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 60)
        self.hidden_fc = nn.Linear(60, 25)
        self.output_fc = nn.Linear(25, output_dim)

    def forward(self, x):
        # x = [batch size, height, width]

        batch_size = x.shape[0]

        # x = x.view(batch_size, -1)

        # x = [batch size, height * width]

        h_1 = F.relu(self.input_fc(x))

        # h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))

        # h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)

        # y_pred = [batch size, output dim]

        return y_pred, h_2
