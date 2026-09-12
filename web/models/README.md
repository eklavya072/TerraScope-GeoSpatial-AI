# Models served by the demo

These are the graphs `web/demo.html` runs in the browser, and the only copies of
them in the repository. They are byte-identical to the graphs the benchmark
measured.

The full export (75 graphs: 5 architectures x 5 seeds x 3 precisions, ~1.2 GB)
is not committed. It is regenerable with `make export` from trained checkpoints,
and no result depends on having it in the repository.

Seven graphs are committed: the five int8-static models the demo races, plus
the fp32 parents of the two small enough to ship. The configuration the
benchmark recommends for CPU-only deployment, and its fp32 parent:

| File | Accuracy (test fold) | Energy/1k | p95 latency | Size |
|---|---|---|---|---|
| `efficientnet_lite0_seed0_int8_static.onnx` | 97.45% ± 0.32 (mean over 5 seeds) | 3.18 J | 0.43 ms | 3.8 MB |
| `efficientnet_lite0_seed0_fp32.onnx` | 97.61% ± 0.13 (mean over 5 seeds) | 19.28 J | 3.74 ms | 13.5 MB |

Both take a normalised NCHW float32 tensor, shape (batch, 3, 64, 64), using
ImageNet channel statistics: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225).
Outputs are raw logits over the 10 EuroSAT classes in the order listed in
`bench/config.py`.

Run them through ONNX Runtime's CPU execution provider. Measured figures above
apply to one intra-op thread on the hardware stated in the top-level README;
they are properties of the model AND that hardware.
