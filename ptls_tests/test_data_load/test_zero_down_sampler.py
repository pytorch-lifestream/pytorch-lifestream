import pytest
from ptls.data_load import ZeroDownSampler


@pytest.mark.parametrize("targets", [[1, 0, 1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0]])
def test_sampler_balances_labels_without_repeating_indices(targets):
    sampler = ZeroDownSampler(targets)
    indices = list(sampler)
    assert len(indices) == len(sampler) == 2 * sum(targets)
    assert len(indices) == len(set(indices))
    assert sum(targets[index] for index in indices) == sum(targets)
    assert all(0 <= index < len(targets) for index in indices)
