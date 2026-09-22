# Testing

These instructions let contributors install a known set of dependencies and
check changes to the library. Run the commands from the checkout under review.
Notebook checks, where available, are described after the library checks.

## Install a test environment

The lock files describe two tested configurations for Linux x86-64, using
central processing unit (CPU) builds of PyTorch: Python 3.13 and Python 3.10.
They pin the dependencies needed for the checks below. These configurations do
not change the library's declared minimum Python version.

For Python 3.13:

```bash
python3.13 -m venv .venv/ptls313
source .venv/ptls313/bin/activate
python -m pip install -r requirements/test-py313.lock
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
```

For Python 3.10, use the other checked-in configuration:

```bash
python3.10 -m venv .venv/ptls310
source .venv/ptls310/bin/activate
python -m pip install -r requirements/test-py310.lock
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
```

Keep the selected environment activated. Reuse it while its lock file is
unchanged. If you move to another checkout, run the editable installation
command there again so imports use that checkout. For validation, use a fresh
checkout without previously generated models or notebook outputs.

## Run the library tests

The full suite includes Spark checks. Install Java 17 and make it available on
`PATH`, or set `JAVA_HOME` to its installation. Select the activated Python for
Spark workers and limit local resource use:

```bash
export PYSPARK_PYTHON="$(command -v python)"
export PYSPARK_SUBMIT_ARGS='--master local[1] --conf spark.sql.shuffle.partitions=2 --conf spark.default.parallelism=2 pyspark-shell'
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
pytest
```

A successful run ends with a passed test summary and exit code zero. Failures
include the test name and traceback. Some tests download data, so this is not
an offline check. Add `--junitxml=.test-results/library.xml` to save a report.

## Notebook tooling

The environment includes nbmake, a pytest plugin that executes notebook cells.
The existing notebooks still need compatibility fixes; the CoLES notebooks do
not yet support reduced test mode. **Use the library tests to validate this
change; notebook execution is not yet a passing check.**

The initial argument file records the encoder-before-fine-tuning order. You can
inspect collection without running either notebook:

```bash
pytest --nbmake @tutorials/notebook-tests/coles.txt --collect-only -q
```

This verifies discovery only. The later notebook changes will provide executable
examples and commands for reduced test runs.

## Update dependencies

The `.in` files list direct test dependencies; `setup.py` lists library dependencies.
The `.lock` files pin the complete resolved set and are installed with pip.
When changing a lock, check a fresh installation without system site packages;
otherwise undeclared packages can hide missing dependencies.

To regenerate a lock, use uv 0.12.17 with the matching Python interpreter. For
Python 3.10, the command used to produce the checked-in file is:

```bash
uv pip compile setup.py requirements/test-py310.in \
  --python .venv/ptls310/bin/python \
  --default-index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match --emit-index-url \
  --no-annotate --no-header --format requirements.txt \
  --output-file requirements/test-py310.lock
```

Use `test-py313.in`, `test-py313.lock` and `.venv/ptls313/bin/python` for the other
configuration. Install the regenerated lock and rerun the relevant checks.
