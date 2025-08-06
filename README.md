# pyRw

Fast multihistogram reweighting in Python
using `numba` vectorisation and jit compilation.

## Installation

To install `pyRw`
from the project home directory type:

```bash
pip install -e .
```

To install the `dev` version,
which has further requirements:

```bash
pip install -e '.[dev]'
```

## Usage

`pyRw` contains a few core functions,
found in `pyRw/core.py`, which are
mostly vectorised or jit compiled by `numba`.
This leaves a few sharp edges in the code,
which is envisioned to be used via the
`MultiRw` interface class from `pyRw/mrw.py`.

## Testing

Testing of the code is implemented in
[`pytest`](https://docs.pytest.org/en/stable/).
To run the tests from the project home directory:

```bash
pytest
```
