import importlib.util
import os
import pickle
from pathlib import Path
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from ptls.preprocessing import PandasDataPreprocessor


def make_preprocessor():
    return PandasDataPreprocessor(
        col_id="id", col_event_time="dt", cols_category=["category"],
        cols_numerical=["amount"], cols_identity=["label"], n_jobs=1,
    )


def sample_data():
    return pd.DataFrame({
        "id": [1, 1, 2], "dt": ["2020-01-02", "2020-01-01", "2020-01-03"],
        "category": ["a", "b", "a"], "amount": [20., 10., 30.],
        "label": [101, 100, 200],
    })


@pytest.mark.parametrize("fitted", [False, True])
def test_pickle_roundtrip_keeps_transformers_and_recreates_dispatcher(fitted, preprocessing_dask_server):
    preprocessor = make_preprocessor()
    if fitted:
        preprocessor.fit_transform(sample_data())
    dispatcher = preprocessor.multithread_dispatcher
    restored = pickle.loads(pickle.dumps(preprocessor))
    assert preprocessor.multithread_dispatcher is dispatcher
    assert restored.multithread_dispatcher is None
    if fitted:
        assert restored.get_category_dictionary_sizes() == preprocessor.get_category_dictionary_sizes()
        assert restored.cts_category[0].mapping == preprocessor.cts_category[0].mapping
    if not fitted:
        restored.fit(sample_data())
    for _ in range(2):
        records = {row["id"]: row for row in restored.transform(sample_data())}
        assert records[1]["event_time"].tolist() == [1577836800, 1577923200]
        assert records[1]["amount"].tolist() == [10., 20.]
        assert records[1]["label"].tolist() == [100, 101]


def run_without_dependencies(blocked, body):
    # Use a new interpreter so prior test imports cannot hide eager dependencies.
    source_root = Path(importlib.util.find_spec("ptls").origin).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    script = textwrap.dedent('''
        import importlib.abc
        import sys

        class BlockDependencies(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + ".") for name in BLOCKED):
                    raise ModuleNotFoundError("Dependency intentionally unavailable: " + fullname, name=fullname)

        sys.meta_path.insert(0, BlockDependencies())
    ''').replace("BLOCKED", repr(blocked)) + textwrap.dedent(body)
    result = subprocess.run([sys.executable, "-c", script], env=env, cwd=source_root,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_pandas_construction_does_not_require_spark_or_distributed():
    run_without_dependencies(["pyspark", "distributed", "dask.distributed"], '''
        import pickle
        from ptls.preprocessing import PandasDataPreprocessor
        from ptls.preprocessing.dask.dask_client import DaskServer

        pp = PandasDataPreprocessor(col_id="id", col_event_time="event_time",
            event_time_transformation="none", cols_category=[], n_jobs=1)
        assert pp.multithread_dispatcher is None
        pickle.loads(pickle.dumps(pp))
        assert not any(name == "pyspark" or name.startswith("distributed.") for name in sys.modules)
    ''')


def test_pandas_execution_does_not_require_spark():
    run_without_dependencies(["pyspark"], '''
        import pandas as pd
        from ptls.preprocessing import PandasDataPreprocessor

        pp = PandasDataPreprocessor(col_id="id", col_event_time="event_time",
            event_time_transformation="none", cols_category=[], n_jobs=1)
        try:
            result = pp.fit_transform(pd.DataFrame({"id": [1, 1], "event_time": [2, 1]}))
            assert result[0]["event_time"].tolist() == [1, 2]
        finally:
            client = pp.multithread_dispatcher.dask_client
            cluster = client.cluster
            try:
                client.close()
            finally:
                cluster.close()
    ''')


def test_spark_construction_does_not_require_distributed():
    run_without_dependencies(["distributed", "dask.distributed"], '''
        from ptls.preprocessing import PysparkDataPreprocessor
        pp = PysparkDataPreprocessor(col_id="id", col_event_time="event_time",
            event_time_transformation="none", cols_identity=["label"])
        assert pp.multithread_dispatcher is None
    ''')


def test_missing_backend_dependency_is_reported_when_requested():
    run_without_dependencies(["pyspark"], '''
        import ptls.preprocessing as preprocessing
        try:
            preprocessing.PysparkDataPreprocessor
        except ModuleNotFoundError as exc:
            assert exc.name == "pyspark"
        else:
            raise AssertionError("Missing Spark must not produce a placeholder backend")
    ''')
