# Pytest Unit Tests

Run the fast unit-test gate before long experiments:

```powershell
python -m pytest unittest
```

The tests cover the YAML dispatcher, config parsing, experiment-suite expansion,
MAD visualization, synthetic dataset windows, model/attention shapes, score
helpers, and statistical baseline helpers.

Tests that need PyTorch use `pytest.importorskip("torch")`. In a lightweight
Python environment without PyTorch they are skipped; in the training environment
they run normally.
