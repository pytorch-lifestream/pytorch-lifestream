"""Keep category semantics stable when reusing a trained sequence encoder."""
import pickle

import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError

from ptls.preprocessing import PandasDataPreprocessor
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.data_load.datasets import inference_data_loader

pytestmark = pytest.mark.usefixtures("preprocessing_dask_server")


def frame(categories):
    return pd.DataFrame({'id': [1]*len(categories), 'time': range(len(categories)),
                         'cat': categories, 'amount': [1.0]*len(categories)})


def preprocessor():
    return PandasDataPreprocessor('id', 'time', event_time_transformation='none',
                                 cols_category=['cat'], cols_numerical=['amount'])


@pytest.mark.parametrize('method', ['fit', 'fit_transform'])
@pytest.mark.parametrize('restore', [False, True])
def test_transform_reuses_fitted_mapping_with_changed_frequencies(method, restore):
    pp = preprocessor()
    getattr(pp, method)(frame(['a', 'a', 'a', 'b']))
    before = pp.cts_category[0].mapping.copy()
    sizes = pp.get_category_dictionary_sizes()
    if restore:
        pp = pickle.loads(pickle.dumps(pp))
    for _ in range(2):
        result = pp.transform(frame(['b', 'b', 'new', 'a']))
        assert result[0]['cat'].tolist() == [before['b'], before['b'], 3, before['a']]
        assert pp.cts_category[0].mapping == before
        assert pp.get_category_dictionary_sizes() == sizes
    # Explicit fitting is still allowed after inference.
    pp.fit_transform(frame(['b', 'b', 'a']))
    assert pp.cts_category[0].mapping == {'b': 1, 'a': 2}


def test_transform_requires_fitted_category_encoder():
    with pytest.raises(NotFittedError):
        preprocessor().transform(frame(['a', 'b']))


def test_failed_transform_does_not_change_dispatcher_mode():
    pp = preprocessor()
    pp.fit_transform(frame(['a', 'a', 'b']))
    before = pp.multithread_dispatcher.transformation_func
    with pytest.raises(KeyError):
        pp.transform(frame(['b']).drop(columns='cat'))
    assert pp.multithread_dispatcher.transformation_func == before
    pp.fit_transform(frame(['c', 'c', 'd']))
    assert pp.cts_category[0].mapping == {'c': 1, 'd': 2}


def test_saved_encoder_and_preprocessor_preserve_embeddings(tmp_path):
    pp = preprocessor()
    pp.fit_transform(frame(['a', 'a', 'b']))
    torch.manual_seed(42)
    def encoder():
        return RnnSeqEncoder(trx_encoder=TrxEncoder(
            embeddings={'cat': {'in': 4, 'out': 4}}, numeric_values={'amount': 'identity'}),
            hidden_size=8, type='gru').eval()
    model = encoder()
    path = tmp_path/'encoder.pt'
    torch.save(model.state_dict(), path)
    restored = encoder()
    restored.load_state_dict(torch.load(path, map_location='cpu'))
    restored_pp = pickle.loads(pickle.dumps(pp))
    held_out = frame(['b', 'new', 'a'])
    records = restored_pp.transform(held_out)
    # Construct expected codes independently, using the training vocabulary.
    expected = dict(records[0], cat=torch.tensor([2, 3, 1]))
    actual_batch = next(iter(inference_data_loader(records)))
    expected_batch = next(iter(inference_data_loader([expected])))
    with torch.no_grad():
        torch.testing.assert_close(restored(actual_batch), model(expected_batch))
