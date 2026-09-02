# `app/` — assets for the demo

Everything here supports the Streamlit demo at the repository root
(`app_demo.py`). There is one app in this repository.

| Path | What it is |
|---|---|
| `demo_data.py` | Reads the committed benchmark artefacts and runs live ONNX inference. No Streamlit, so it is unit-testable. |
| `models_onnx/` | The 15 seed-0 ONNX graphs the demo serves (5 architectures × 3 precisions), byte-identical to the graphs the benchmark measured. |
| `samples/` | 20 demo tiles, 2 per class, **all from the held-out test fold** of the committed split, with `index.json` recording each tile's true label, its path in the split, and the split's sha256. |

## The TensorFlow demo that used to live here

The original project was a single-model Keras demo. It has been moved to the
`archive/legacy-tensorflow-app` branch and removed from `master`, because:

- it carried 317 MB of `.h5` weights in Git LFS and could not run inside the
  demo's memory budget;
- it was trained on a third-party split whose training fold overlaps 1,949 of
  the 2,700 tiles in this benchmark's test fold, so its 95.67% figure is not
  comparable to anything published here;
- keeping two apps in one repository invites exactly the confusion this
  benchmark exists to avoid.

```bash
git checkout archive/legacy-tensorflow-app   # if you want to see it
```
