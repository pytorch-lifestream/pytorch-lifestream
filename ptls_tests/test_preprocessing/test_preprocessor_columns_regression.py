import pandas as pd
import pytest
import torch

from ptls.preprocessing.pandas.pandas_preprocessor import PandasDataPreprocessor
from ptls.preprocessing.pyspark.pyspark_preprocessor import PysparkDataPreprocessor
from ptls.preprocessing.pyspark.col_identity_transformer import ColIdentityEncoder as SparkIdentity


@pytest.mark.parametrize("categories", [None, [], ["category"]])
@pytest.mark.parametrize("vector", [False, True])
def test_empty_categories_with_identity_columns(categories, vector, preprocessing_dask_server):
    labels = ([torch.tensor([100., 101.]), torch.tensor([102., 103.]), torch.tensor([200., 201.])]
              if vector else [100, 101, 200])
    data = pd.DataFrame({"id": [1, 1, 2], "event_time": [2, 1, 1],
        "amount": [20., 10., 30.], "category": ["a", "b", "a"], "labels": labels})
    pp = PandasDataPreprocessor(col_id="id", col_event_time="event_time",
        event_time_transformation="none", cols_category=categories,
        cols_numerical=["amount"], cols_identity=["labels"], n_jobs=1)
    for operation in (pp.fit_transform, pp.transform):
        features = {record["id"]: record for record in operation(data)}
        assert len(features) == 2
        expected = [[102., 103.], [100., 101.]] if vector else [101, 100]
        assert features[1]["labels"].tolist() == expected
        assert features[1]["amount"].tolist() == [10., 20.]
        assert features[1]["event_time"].tolist() == [1, 2]


@pytest.mark.parametrize("categories", [None, []])
def test_event_time_only_preprocessing(categories, preprocessing_dask_server):
    pp = PandasDataPreprocessor(col_id="id", col_event_time="event_time",
        event_time_transformation="none", cols_category=categories, n_jobs=1)
    records = pp.fit_transform(pd.DataFrame({"id": [1, 1], "event_time": [2, 1]}))
    assert records[0]["event_time"].tolist() == [1, 2]
    assert pp.get_category_dictionary_sizes() == {}


def test_spark_uses_its_own_identity_transformer():
    pp = PysparkDataPreprocessor(col_id="id", col_event_time="event_time",
        event_time_transformation="none", cols_identity=["labels"])
    assert isinstance(pp.cts_identity[0], SparkIdentity)
