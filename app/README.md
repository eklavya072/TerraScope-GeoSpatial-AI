# `app/` — demo tile provenance

This directory is the record of which EuroSAT tiles the demo shows and where
they came from. It is not an application; the demo is the static site in
`web/`.

| Path | What it is |
|---|---|
| `samples/index.json` | Per tile: its true label, its path in the split, the source file's sha256, the sha256 of the split the selection was drawn from, and which models classify it correctly. |

`scripts/pick_demo_tiles.py` writes this record and the images the site serves,
which live in `web/assets/` — one copy, not two. The record does not duplicate
the images because what it attests to is the sha256 of the **original corpus
file**, which a re-saved PNG could not vouch for anyway. `scripts/build_site_data.py` reads
`samples/index.json` when it exports `web/data/site.json`, so the labels the
site shows are the ones recorded here rather than typed into the markup.

Keeping the record separate from the served copies is the point: a reader can
check that every tile on the demo is one no model was trained on, without
trusting the site's own copy of the answer.

## Earlier apps that lived here

Two demos preceded the static site, both removed rather than left to rot beside
it:

- a **Streamlit** app, replaced by `web/`, which needs no Python process to
  serve and no server-side inference;
- the original **TensorFlow/Keras** demo, moved to the
  `archive/legacy-tensorflow-app` branch. It carried 317 MB of `.h5` weights in
  Git LFS, and its 95.67% was measured on a third-party split whose training
  fold overlaps 1,949 of the 2,700 tiles in this benchmark's test fold, so that
  figure is not comparable to anything published here.

```bash
git checkout archive/legacy-tensorflow-app   # if you want to see it
```

The ONNX graphs the browser runs live in `web/models/`, and the runtime that
runs them in `web/vendor/ort/`. There is one copy of each.
