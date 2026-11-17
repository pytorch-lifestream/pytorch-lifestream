"""Preserve distinct outputs when column transformations share an input."""

import pickle

import pandas as pd
import pytest

from ptls.preprocessing import PandasDataPreprocessor
from ptls.preprocessing.base.transformation.col_numerical_transformer import ColTransformer
from ptls.preprocessing.pandas.pandas_transformation.pandas_freq_transformer import FrequencyEncoder

pytestmark = pytest.mark.usefixtures("preprocessing_dask_server")


class DoubleValues(ColTransformer):
    def transform(self, frame):
        values = (frame[self.col_name_original] * 2).rename(self.col_name_target)
        return self.drop_original_col(self.attach_column(frame, values))


class CenterValues(ColTransformer):
    def fit(self, frame):
        self.center_ = frame[self.col_name_original].mean()
        return self

    def transform(self, frame):
        values = (frame[self.col_name_original] - self.center_).rename(self.col_name_target)
        return self.drop_original_col(self.attach_column(frame, values))


def fitted_outputs(preprocessor, source, method, restore=False):
    original = source.copy(deep=True)
    result = getattr(preprocessor, method)(source)
    if method == "fit":
        assert result is preprocessor
    results = [result] if method == "fit_transform" else []
    if restore:
        preprocessor = pickle.loads(pickle.dumps(preprocessor))
    results.append(preprocessor.transform(source))
    pd.testing.assert_frame_equal(source, original)
    return results


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("restore", [False, True])
def test_datetime_identity_keeps_raw_dates_and_timestamps(method, restore):
    source = pd.DataFrame({
        "id": [1, 1, 1],
        "dt": ["2020-01-03", "2020-01-01", "2020-01-02"],
    })
    preprocessor = PandasDataPreprocessor("id", "dt", cols_identity=["dt"], n_jobs=1)

    for records in fitted_outputs(preprocessor, source, method, restore):
        assert records[0]["dt"].tolist() == ["2020-01-01", "2020-01-02", "2020-01-03"]
        assert records[0]["event_time"].tolist() == [1577836800, 1577923200, 1578009600]


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("restore", [False, True])
def test_shared_numerical_outputs_keep_raw_values_and_fitted_state(method, restore):
    source = pd.DataFrame({"id": [1, 1, 1], "event_time": [3, 1, 2], "amount": [9., 4., 5.]})
    preprocessor = PandasDataPreprocessor(
        "id", "event_time", event_time_transformation="none",
        cols_numerical=[CenterValues("amount", "centered"), DoubleValues("amount", "scaled")],
        cols_identity=["amount"], n_jobs=1,
    )

    for records in fitted_outputs(preprocessor, source, method, restore):
        assert records[0]["event_time"].tolist() == [1, 2, 3]
        assert records[0]["amount"].tolist() == [4., 5., 9.]
        assert records[0]["centered"].tolist() == [-2., -1., 3.]
        assert records[0]["scaled"].tolist() == [8., 10., 18.]
    if restore:
        preprocessor = pickle.loads(pickle.dumps(preprocessor))
    held_out = source.assign(amount=source["amount"] + 10)
    record = preprocessor.transform(held_out)[0]
    assert record["amount"].tolist() == [14., 15., 19.]
    assert record["centered"].tolist() == [8., 9., 13.]
    assert record["scaled"].tolist() == [28., 30., 38.]


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_numeric_event_time_alias_keeps_requested_source(method):
    source = pd.DataFrame({"id": [1, 1], "time": [20., 10.]})
    preprocessor = PandasDataPreprocessor(
        "id", "time", event_time_transformation="none", cols_identity=["time"], n_jobs=1,
    )
    for records in fitted_outputs(preprocessor, source, method):
        assert records[0]["time"].tolist() == [10., 20.]
        assert records[0]["event_time"].tolist() == [10., 20.]


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("kind, expected", [("numerical", [8., 8., 18.]), ("category", [1, 1, 2])])
def test_identity_does_not_overwrite_explicit_inplace_output(method, restore, kind, expected):
    source = pd.DataFrame({"id": [1, 1, 1], "event_time": [3, 1, 2], "value": [9., 4., 4.]})
    transformer = DoubleValues("value") if kind == "numerical" else FrequencyEncoder("value")
    preprocessor = PandasDataPreprocessor(
        "id", "event_time", event_time_transformation="none", cols_identity=["value"],
        n_jobs=1, **{f"cols_{kind}": [transformer]},
    )

    for records in fitted_outputs(preprocessor, source, method, restore):
        assert records[0]["value"].tolist() == expected


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("custom_first", [False, True])
def test_category_names_and_transformers_can_be_mixed(method, custom_first):
    source = pd.DataFrame({
        "id": [1, 1, 1], "event_time": [3, 1, 2],
        "cat_a": ["a", "a", "b"], "cat_b": ["x", "y", "y"],
    })
    categories = [FrequencyEncoder("cat_a", col_name_target="encoded_a"), "cat_b"]
    if not custom_first:
        categories.reverse()
    preprocessor = PandasDataPreprocessor(
        "id", "event_time", event_time_transformation="none", cols_category=categories, n_jobs=1,
    )

    for records in fitted_outputs(preprocessor, source, method):
        assert records[0]["encoded_a"].tolist() == [1, 2, 1]
        assert records[0]["cat_b"].tolist() == [1, 1, 2]
        assert "cat_a" not in records[0]


def test_restored_preprocessor_accepts_legacy_dispatcher_keys():
    source = pd.DataFrame({
        "id": [1, 1, 1, 1], "event_time": [1, 2, 3, 4], "cat": ["a", "a", "a", "b"],
    })
    preprocessor = PandasDataPreprocessor(
        "id", "event_time", event_time_transformation="none", cols_category=["cat"], n_jobs=1,
    )
    preprocessor.fit_transform(source)
    # Earlier fitted preprocessors serialized this cache under source-column names.
    preprocessor.unitary_func = {
        transformer.col_name_original: transformer
        for transformer in preprocessor.unitary_func.values()
    }
    restored = pickle.loads(pickle.dumps(preprocessor))
    held_out = source.assign(cat=["b", "b", "new", "a"])

    assert restored.transform(held_out)[0]["cat"].tolist() == [2, 2, 3, 1]
    assert restored.get_category_dictionary_sizes() == {"cat": 4}
