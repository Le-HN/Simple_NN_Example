import numpy as np
import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_dim),
        )

        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax_layer(x)

        return x


if __name__ == "__main__":
    net = NN(10, 2)
    test = torch.as_tensor(np.array([1, 2, 1, 0, 1, 0, 2, 1, 2, 0]), dtype=torch.float32)
    out = net.forward(test)
    print(out)