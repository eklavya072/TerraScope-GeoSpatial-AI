# Legacy Streamlit demo

This is the project's original single-model demo: upload a satellite tile, get a
land-cover prediction and a confidence chart. It predates the benchmark and is
kept because it still runs, not because any benchmark result depends on it.

**It is not part of the benchmark.** It uses TensorFlow/Keras and its own
dependency set, deliberately isolated from the benchmark environment so that
TensorFlow can never load inside a measured inference run.

```bash
pip install -r app/requirements-app.txt
streamlit run app/app.py          # run from the REPOSITORY ROOT
```

## About its reported accuracy

The demo's model reports **95.67%** test accuracy. That figure was independently
re-measured during the benchmark work and reproduced exactly, so it is real —
but it was measured against a third-party split whose training fold overlaps
**1,949 of the 2,700 tiles** in the benchmark's test fold. It is therefore not
comparable to any number in the benchmark and is not carried forward.

The benchmark retrains from scratch against a committed, hash-identified split;
see the top-level README.
