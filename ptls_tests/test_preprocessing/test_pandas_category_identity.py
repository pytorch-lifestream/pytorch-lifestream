"""Respect explicit category identity encoding through the public preprocessor."""
import pickle

import pandas as pd
import pytest
import torch

from ptls.preprocessing import PandasDataPreprocessor

pytestmark = pytest.mark.usefixtures("preprocessing_dask_server")


@pytest.mark.parametrize('method', ['fit', 'fit_transform'])
@pytest.mark.parametrize('restore', [False, True])
def test_identity_keeps_sparse_category_codes(method, restore):
    source = pd.DataFrame({'id': [1, 1, 1], 'time': [3, 1, 2], 'cat': [9, 4, 4]})
    pp = PandasDataPreprocessor('id', 'time', event_time_transformation='none',
                               cols_category=['cat'], category_transformation='none')
    result = getattr(pp, method)(source)
    if method == 'fit_transform':
        torch.testing.assert_close(result[0]['cat'], torch.tensor([4, 4, 9]))
    if restore:
        pp = pickle.loads(pickle.dumps(pp))
    result = pp.transform(source)
    torch.testing.assert_close(result[0]['cat'], torch.tensor([4, 4, 9]))
    assert pp.get_category_dictionary_sizes() == {'cat': 10}


def test_default_frequency_encoding_is_unchanged():
    source = pd.DataFrame({'id': [1, 1, 1], 'time': [3, 1, 2], 'cat': [9, 4, 4]})
    pp = PandasDataPreprocessor('id', 'time', event_time_transformation='none', cols_category=['cat'])
    torch.testing.assert_close(pp.fit_transform(source)[0]['cat'], torch.tensor([1, 1, 2]))
