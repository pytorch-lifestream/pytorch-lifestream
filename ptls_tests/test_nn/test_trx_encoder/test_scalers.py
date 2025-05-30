import torch

from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder import TrxEncoder
from ptls.nn.trx_encoder.encoders import BaseEncoder

from ptls.nn.trx_encoder.scalers import (
    Periodic, PeriodicMLP, PLE, PLE_MLP,
    IdentityScaler, SigmoidScaler, LogScaler, YearScaler,
    NumToVector, LogNumToVector, PoissonScaler, ExpScaler
)


def test_periodic():
    B, T = 5, 20
    num_periods = 4
    scaler = Periodic(num_periods = num_periods, param_dist_sigma = 3)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, 2 * num_periods)  # B, T, H
    assert trx_encoder.output_size == 2 * num_periods


def test_periodic_mlp():
    B, T = 5, 20
    num_periods = 4
    scaler = PeriodicMLP(num_periods = num_periods, param_dist_sigma = 3)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, 2 * num_periods)  # B, T, H
    assert trx_encoder.output_size == 2 * num_periods


def test_periodic_mlp2():
    B, T = 5, 20
    num_periods = 4
    mlp_output_size = 32
    scaler = PeriodicMLP(num_periods = num_periods, param_dist_sigma = 3, mlp_output_size = mlp_output_size)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, mlp_output_size)  # B, T, H
    assert trx_encoder.output_size == mlp_output_size


def test_ple():
    B, T = 5, 20
    bins = [-1, 0, 1]
    scaler = PLE(bins = bins)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, len(bins) - 1)  # B, T, H
    assert trx_encoder.output_size == len(bins) - 1


def test_ple2():
    B, T = 1, 20
    bins = [-1, 0, 1]
    scaler = PLE(bins = bins)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
        use_batch_norm=False,
    )
    x = PaddedBatch(
        payload={
            'amount': torch.tensor([[-2, -1, 0, 1, 2, -0.2, 0.7]]),
        },
        length=torch.tensor([7]),
    )
    z = trx_encoder(x)
    ans = torch.tensor([[
         [0., 0.],
         [0., 0.],
         [1., 0.],
         [1., 1.],
         [1., 1.],
         [0.8, 0.0],
         [1., 0.7]
    ]])
    assert torch.abs(z.payload - ans).sum() < 1e-5, z.payload


def test_ple_mlp():
    B, T = 5, 20
    bins = [-1, 0, 1]
    scaler = PLE_MLP(bins = bins)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, len(bins) - 1)  # B, T, H
    assert trx_encoder.output_size == len(bins) - 1


def test_ple_mlp2():
    B, T = 5, 20
    bins = [-1, 0, 1]
    mlp_output_size = 64
    scaler = PLE_MLP(bins = bins, mlp_output_size = mlp_output_size)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (5, 20, mlp_output_size)  # B, T, H
    assert trx_encoder.output_size == mlp_output_size


def test_identity_scaler():
    B, T = 5, 20
    trx_encoder = TrxEncoder(
        numeric_values={'amount': 'identity'},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (B, T, 1)
    assert trx_encoder.output_size == 1


def test_sigmoid_scaler():
    B, T = 5, 20
    trx_encoder = TrxEncoder(
        numeric_values={'amount': 'sigmoid'},
        use_batch_norm=False,
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (B, T, 1)
    assert trx_encoder.output_size == 1


def test_log_scaler():
    B, T = 5, 20
    trx_encoder = TrxEncoder(
        numeric_values={'amount': 'log'},
        use_batch_norm=False,
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (B, T, 1)
    assert trx_encoder.output_size == 1


def test_year_scaler():
    B, T = 5, 20
    trx_encoder = TrxEncoder(
        numeric_values={'amount': 'year'},
        use_batch_norm=False,
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T) * 365,  # Генерируем значения в днях
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (B, T, 1)
    assert trx_encoder.output_size == 1


def test_num_to_vector():
    B, T = 5, 20
    embeddings_size = 32
    scaler = NumToVector(embeddings_size)   
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (B, T, embeddings_size)
    assert trx_encoder.output_size == embeddings_size


def test_log_num_to_vector():
    B, T = 5, 20
    embeddings_size = 32
    scaler = LogNumToVector(embeddings_size)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (B, T, embeddings_size)
    assert trx_encoder.output_size == embeddings_size


def test_poisson_scaler():
    B, T = 5, 20
    scaler = PoissonScaler(kmax=33)
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
    )
    x = PaddedBatch(
        payload={
            'amount': torch.rand(B, T) * 10,  # Генерируем положительные значения
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (B, T, 1)
    assert trx_encoder.output_size == 1


def test_exp_scaler():
    B, T = 5, 20
    scaler = ExpScaler()
    trx_encoder = TrxEncoder(
        numeric_values={'amount': scaler},
        use_batch_norm=False,
    )
    x = PaddedBatch(
        payload={
            'amount': torch.randn(B, T),
        },
        length=torch.randint(10, 20, (B,)),
    )
    z = trx_encoder(x)
    assert z.payload.shape == (B, T, 1)
    assert trx_encoder.output_size == 1
    assert torch.all(z.payload > 0)