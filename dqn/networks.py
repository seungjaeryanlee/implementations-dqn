"""Define necessary neural networks.

To match ReLU activations, we use Kaiming He initialization.
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize Q-Network.

        Parameters
        ----------
        in_dim : int
            Dimension of the input layer.
        out_dim : int
            Dimension of the output layer.

        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.layers.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the network. Should be the observation.

        Returns
        -------
        q_values : torch.Tensor
            Output tensor of the network. Q-values of all actions.

        """
        return self.layers(x)


class AtariQNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize Q-Network for Atari environments.

        Parameters
        ----------
        in_dim : int
            Number of stacked frames.
        out_dim : int
            Number of actions.

        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_dim, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, out_dim)
        )

        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.conv_layers.apply(weights_init)
        self.fc_layers.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the network. Should be the observation.

        Returns
        -------
        q_values : torch.Tensor
            Output tensor of the network. Q-values of all actions.

        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x
