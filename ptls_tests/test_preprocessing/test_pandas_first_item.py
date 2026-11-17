"""Preserve scalar user-level fields through the public Pandas preprocessor."""
import pickle

import pandas as pd
import pytest
import torch

from ptls.preprocessing import PandasDataPreprocessor

pytestmark = pytest.mark.usefixtures("preprocessing_dask_server")


def transactions():
    # First values must follow event-time order, not the input row order.
    return pd.DataFrame({'user_id': [2, 1, 2, 1], 'time': [5, 3, 1, 1],
                         'category': ['b', 'a', 'b', 'a'],
                         'amount': [2.5, 1.3, 2.1, 1.1],
                         'target': [25, 13, 21, 11]})


@pytest.mark.parametrize('return_records', [False, True])
@pytest.mark.parametrize('method', ['fit', 'fit_transform'])
@pytest.mark.parametrize('restore', [False, True])
def test_first_item_preserves_scalar_targets(return_records, method, restore):
    columns = ['target']
    pp = PandasDataPreprocessor(
        'user_id', 'time', event_time_transformation='none',
        cols_category=['category'], cols_numerical=['amount'],
        cols_identity=['target'], cols_first_item=columns,
        return_records=return_records,
    )
    result = getattr(pp, method)(transactions())
    mapping = pp.cts_category[0].mapping.copy()
    if restore:
        pp = pickle.loads(pickle.dumps(pp))
    for result in [result if method == 'fit_transform' else pp.transform(transactions()),
                   pp.transform(transactions())]:
        rows = result if return_records else result.to_dict(orient='records')
        assert {row['user_id']: row['target'] for row in rows} == {1: 11, 2: 21}
        assert all(pd.api.types.is_scalar(row['target']) for row in rows)
        torch.testing.assert_close(rows[0]['event_time'], torch.tensor([1, 3]))
        torch.testing.assert_close(rows[1]['event_time'], torch.tensor([1, 5]))
    assert columns == ['target']
    assert pp.cts_category[0].mapping == mapping


@pytest.mark.parametrize('first_columns', [None, []])
def test_default_aggregation_keeps_target_sequence(first_columns):
    pp = PandasDataPreprocessor(
        'user_id', 'time', event_time_transformation='none',
        cols_category=[], cols_identity=['target'], cols_first_item=first_columns,
    )
    rows = pp.fit_transform(transactions())
    torch.testing.assert_close(rows[0]['target'], torch.tensor([11, 13]))
    torch.testing.assert_close(rows[1]['target'], torch.tensor([21, 25]))
