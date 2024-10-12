from torch import nn


class MyModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim, output_dim: int):
        super(MyModel, self).__init__()
        if hidden_dim is int:
            hidden_dim = [hidden_dim]

        layers = []
        in_dim = input_dim
        for dim in hidden_dim:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.Tanh())
            in_dim = dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x