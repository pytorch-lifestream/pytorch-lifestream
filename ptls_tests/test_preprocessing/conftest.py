import pytest


@pytest.fixture(scope="session")
def preprocessing_dask_server():
    from ptls.preprocessing.dask.dask_client import DaskServer

    # Keep the real shared server alive across preprocessors in execution tests.
    server = DaskServer()
    cluster = server.client.cluster
    try:
        yield server
    finally:
        try:
            server.client.close()
        finally:
            cluster.close()
