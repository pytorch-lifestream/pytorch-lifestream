import pandas as pd
import pytest

from ptls.preprocessing.base.transformation.col_event_time_transformer import DatetimeToTimestamp
from ptls.preprocessing.pandas.pandas_preprocessor import PandasDataPreprocessor


@pytest.mark.parametrize("drop_original", [False, True])
@pytest.mark.parametrize("column", ["dt", "event_time"])
def test_datetime_transformer_preserves_other_columns(drop_original, column):
    data = pd.DataFrame({column: ["1970-01-01", "2020-01-02"], "amount": [1., 2.]}, index=[4, 7])
    original = data.copy(deep=True)
    result = DatetimeToTimestamp(column, is_drop_original_col=drop_original).fit_transform(data)
    assert result["event_time"].tolist() == [0, 1577923200]
    assert result["amount"].tolist() == [1., 2.]
    assert result.index.tolist() == [4, 7]
    if column != "event_time":
        assert (column in result) == (not drop_original)
        if not drop_original:
            pd.testing.assert_series_equal(result[column], original[column])
    pd.testing.assert_frame_equal(data, original)


def test_datetime_transformer_keeps_original_by_default():
    data = pd.DataFrame({"dt": ["2020-01-01"]})
    result = DatetimeToTimestamp("dt").fit_transform(data)
    assert result["dt"].tolist() == ["2020-01-01"]
    assert result["event_time"].tolist() == [1577836800]


def test_pandas_preprocessor_default_datetime_column(preprocessing_dask_server):
    data = pd.DataFrame({
        "id": [1, 1, 2], "dt": ["2020-01-02", "2020-01-01", "2020-01-03"],
        "category": ["a", "b", "a"], "amount": [20., 10., 30.],
    })
    pp = PandasDataPreprocessor(col_id="id", col_event_time="dt",
        cols_category=["category"], cols_numerical=["amount"], n_jobs=1)
    for operation in (pp.fit_transform, pp.transform):
        records = {record["id"]: record for record in operation(data)}
        assert records[1]["event_time"].tolist() == [1577836800, 1577923200]
        assert records[1]["amount"].tolist() == [10., 20.]
        assert "dt" not in records[1]
