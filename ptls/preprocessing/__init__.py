from importlib import import_module

from .pandas.pandas_preprocessor import PandasDataPreprocessor

__all__ = ["PandasDataPreprocessor", "DaskDataPreprocessor", "PysparkDataPreprocessor"]


def __getattr__(name):
    backends = {
        "DaskDataPreprocessor": ".dask.dask_preprocessor",
        "PysparkDataPreprocessor": ".pyspark.pyspark_preprocessor",
    }
    if name not in backends:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    backend = getattr(import_module(backends[name], __name__), name)
    globals()[name] = backend
    return backend
