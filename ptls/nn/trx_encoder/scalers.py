import torch
import torch.nn as nn

from .encoders import BaseEncoder


class IdentityScaler(BaseEncoder):
    def forward(self, x):
        return x.unsqueeze(2).float()

    @property
    def output_size(self):
        return 1


class SigmoidScaler(IdentityScaler):
    def __init__(self, col_name: str = None):
        super().__init__(col_name)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = super().forward(x)
        return self.sigmoid(x)


class LogScaler(IdentityScaler):
    def forward(self, x):
        x = super().forward(x)
        return x.abs().log1p() * x.sign()


class YearScaler(IdentityScaler):
    def forward(self, x):
        x = super().forward(x)
        return x / 365


class NumToVector(IdentityScaler):
    def __init__(self, embeddings_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)

    def forward(self, x):
        x = super().forward(x)
        return x * self.w + self.b

    @property
    def output_size(self):
        return self.w.size(2)


class LogNumToVector(IdentityScaler):
    def __init__(self, embeddings_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)

    def forward(self, x):
        x = super().forward(x)
        return x.abs().log1p() * x.sign() * self.w + self.b

    @property
    def output_size(self):
        return self.w.size(2)


class PoissonScaler(IdentityScaler):
    """
    Explicit estimator for poissonian target with standard pytorch sampler extrapolation.
    """
    def __init__(self, kmax=33):
        super().__init__()
        self.kmax = 0.7 * kmax
        self.arange = torch.nn.Parameter(torch.arange(kmax), requires_grad=False)
        self.factor = torch.nn.Parameter(torch.special.gammaln(1 + self.arange), requires_grad=False)

    def forward(self, x):
        x = super().forward(x)
        if self.kmax == 0:
            return torch.poisson(x)
        res = self.arange * torch.log(x).unsqueeze(-1) - self.factor * torch.ones_like(x).unsqueeze(-1)
        return res.argmax(dim=-1).float().where(x < self.kmax, torch.poisson(x))


class ExpScaler(IdentityScaler):
    def forward(self, x):
        x = super().forward(x)
        return torch.exp(x)


class Periodic(IdentityScaler):
    """
    x -> [cos(cx), sin(cx)], c - (num_periods)-dimensional learnable vector initialized from N(0, param_dist_sigma)

    From paper  "On embeddings for numerical features in tabular deep learning"
    """
    def __init__(self, num_periods=8, param_dist_sigma=1):
        super().__init__()
        self.num_periods = num_periods
        self.c = torch.nn.Parameter(torch.normal(0, param_dist_sigma, size=(1, 1, num_periods)), requires_grad=True)

    def forward(self, x):
        x = super().forward(x)
        x = self.c * x
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=2)
        return x

    @property
    def output_size(self):
        return 2 * self.num_periods


class PeriodicMLP(IdentityScaler):
    """
    x -> [cos(cx), sin(cx)], c - (num_periods)-dimensional learnable vector initialized from N(0, param_dist_sigma)
    Then Linear, then ReLU

    From paper  "On embeddings for numerical features in tabular deep learning"
    """
    def __init__(self, num_periods=8, param_dist_sigma=1, mlp_output_size=-1):
        super().__init__()
        self.num_periods = num_periods
        self.mlp_output_size = mlp_output_size if mlp_output_size > 0 else 2 * self.num_periods
        self.c = torch.nn.Parameter(torch.normal(0, param_dist_sigma, size=(1, 1, num_periods)), requires_grad=True)
        self.mlp = nn.Linear(2 * self.num_periods, self.mlp_output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = super().forward(x)
        x = self.c * x
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=2)
        x = self.mlp(x)
        x = self.relu(x)
        return x

    @property
    def output_size(self):
        return self.mlp_output_size


class PLE(IdentityScaler):
    """
    x -> [1, 1, 1, ax, 0, 0, 0] based on bins
    From paper  "On embeddings for numerical features in tabular deep learning"
    """
    def __init__(self, bins=[-1, 0, 1]):
        super().__init__()
        self.size = len(bins) - 1
        self.bins = torch.tensor([[bins,]])

    def forward(self, x):
        self.bins = self.bins.to(x.device)
        x = super().forward(x)
        x = (x - self.bins[:, :, :-1]) / (self.bins[:, :, 1:] - self.bins[:, :, :-1])
        x = x.clamp(0, 1)
        return x

    @property
    def output_size(self):
        return self.size


class PLE_MLP(IdentityScaler):
    """
    x -> [1, 1, 1, ax, 0, 0, 0] based on bins
    Then Linear, Then ReLU

    From paper  "On embeddings for numerical features in tabular deep learning"
    """
    def __init__(self, bins=[-1, 0, 1], mlp_output_size=-1):
        super().__init__()
        self.size = len(bins) - 1
        self.mlp_output_size = mlp_output_size if mlp_output_size > 0 else self.size
        self.bins = torch.tensor([[bins,]])
        self.mlp = nn.Linear(self.size, self.mlp_output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        self.bins = self.bins.to(x.device)
        x = super().forward(x)
        x = (x - self.bins[:, :, :-1]) / (self.bins[:, :, 1:] - self.bins[:, :, :-1])
        x = x.clamp(0, 1)
        x = self.mlp(x)
        x = self.relu(x)
        return x

    @property
    def output_size(self):
        return self.mlp_output_size


class Time2Vec(IdentityScaler):
    """
    One-dimensional time encoding.
    Proposed in paper "Time2Vec: Learning a Vector Representation of Time".
    Input tensor must contain times of transactions in any units.
    According to the paper, this method should work equally well with absolute times,
    relative times and times since the previous transactions.
    Time values are encoded using linear layers and a cosine function.
    """

    # TODO: implement time normalization

    def __init__(self, embeddings_size: int) -> None:
        """
        Class initialization.

        Args:
            embeddings_size (int): Desired size of periodic time embeddings.
                Total resulting time embeddings will have the size of `embeddings_size` + 1
                because of one extra non-periodic component per transaction.
                This argument is called `k` in the original paper.
        """

        super().__init__()

        self.embeddings_size = embeddings_size

        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, self.embeddings_size)

    def forward(self, event_times: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            event_times (torch.Tensor): Tensor containing absolute or relative event times in any units.

        Returns:
            torch.Tensor: Tensor with resulting time embeddings.
        """

        # Converting the times to the float type and adding an extra dimension at the end
        event_times = event_times.float().unsqueeze(-1)

        # Calculating the non-periodic components
        non_periodic = self.fc1(event_times)

        # Calculating the periodic components
        periodic = torch.cos(self.fc2(event_times))

        # Concatenating the resulting embeddings together
        return torch.cat([non_periodic, periodic], -1)

    @property
    def output_size(self) -> int:
        """
        Returns:
            int: The last dimension of the output tensor.
        """

        return self.embeddings_size + 1


class Time2VecMult(IdentityScaler):
    """
    Multidimensional time encoding.
    Proposed in paper "Incorporating Time in Sequential Recommendation Models" as "Projection-based approach".
    Input tensor must contain Unix timestamps in seconds.
    Timestamps are converted into numerical features and encoded using a linear layer and a cosine function.
    """

    def __init__(self, embeddings_size: int) -> None:
        """
        Class initialization.

        Args:
            embeddings_size (int): Desired size of time embeddings.
        """

        super().__init__()

        self.embeddings_size = embeddings_size

        self.fc = nn.Linear(4, self.embeddings_size)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            timestamps (torch.Tensor): Tensor containing Unix timestamps in seconds.

        Returns:
            torch.Tensor: Tensor with resulting time embeddings.
        """

        # Calculating the numbers of hours and days passed since 01-01-1970
        total_hours = timestamps.float() / 3600.0
        total_days = total_hours / 24.0

        # 1. Normalized hour of the day [0, 1]
        hour_of_day = torch.fmod(total_hours, 24.0) / 24.0

        # 2. Normalized day of the week [0, 1] (0 - Mon, 1 - Sun)
        day_of_week = torch.fmod(total_days + 3.0, 7.0) / 7.0

        # 3. Normalized day of the month [0, 1]
        # (approximation)
        day_of_month = torch.fmod(total_days, 30.44) / 30.44

        # 4. Normalized month of the year [0, 1] (0 - Jan, 1 - Dec)
        # (approximation)
        month_of_year = torch.fmod(total_days / 30.44, 12.0) / 12.0

        # Staking 4 tensors with timestamp features into one
        timestamp_features = torch.stack([
            hour_of_day,
            day_of_week,
            day_of_month,
            month_of_year
        ], dim=-1)

        # Applying a linear layer and a cosine function
        return torch.cos(self.fc(timestamp_features))

    @property
    def output_size(self) -> int:
        """
        Returns:
            int: The last dimension of the output tensor.
        """

        return self.embeddings_size


def scaler_by_name(name):
    scaler = {
        'identity': IdentityScaler,
        'sigmoid': SigmoidScaler,
        'log': LogScaler,
        'year': YearScaler,
        'periodic': Periodic,
        'periodic_mlp': PeriodicMLP,
    }.get(name, None)

    if scaler is None:
        raise Exception(f'unknown scaler name: {name}')
    else:
        return scaler()
