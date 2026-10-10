# Contributing

Questions, bug reports and requests are welcome through [GitHub issues](https://github.com/AmmarKheder/Foehn/issues);
templates are provided for a bug, a question and a data or weights request.

## Running the code

The scripts were run on the LUMI supercomputer (AMD MI250X, ROCm 6.1.3, PyTorch 2.4.1, Python 3.12) inside the container
`lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif`; the SLURM scripts in `foehn/jobs/` show every call. On another
machine, install `requirements.txt` in a PyTorch environment matching your GPU, and set the data paths at the top of the
scripts you run (they are absolute paths on LUMI; search for `/scratch/project_462001140`).

Quick checks that need no data:

```bash
python -m compileall -q foehn figures benchmark_europe        # every script parses
python foehn/tests/test_wellmixed.py                          # FLEXGREEN: the well-mixed limit of the boundary layer
```

## Pull requests

- Keep a change to one purpose, with a short message in the imperative.
- Do not commit data, checkpoints, logs or evaluation outputs (see `.gitignore`); the figure scripts expect their intermediate
  results in `figures/data/`, produced by the `figdata_*.py` scripts.
- Scripts start with a docstring stating what they compute, their inputs and their outputs; keep that convention.
- By contributing you agree that your contribution is released under the MIT licence of the repository.
