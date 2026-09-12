"""Catch the mistakes that tests over the data cannot see.

The suite verifies that every published figure matches the benchmark. It says
nothing about whether a file the markup asks for exists, whether a CSS grid
has room for the things put into it, or whether a page still references a
token that was renamed. Those are the errors that reach the visitor, so they
get their own pass.

    python scripts/check_site.py
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
PAGES = ["index.html", "demo.html", "trade-offs.html", "method.html"]
SCRIPTS = ["site.js", "hero.js", "demo.js", "tradeoffs.js", "method.js"]
CSS = ["site.css", "hero.css"]

fails: list[str] = []


def fail(msg: str) -> None:
    fails.append(msg)


def read(*parts: str) -> str:
    with open(os.path.join(WEB, *parts)) as fh:
        return fh.read()


# --- every local asset a page or stylesheet asks for must exist -------------
for page in PAGES:
    src = read(page)
    for ref in set(re.findall(r'(?:src|href)="(?!https?:|#|mailto:)([^"]+)"', src)):
        target = os.path.join(WEB, ref.split("?")[0])
        if not os.path.exists(target):
            fail(f"{page} references missing file: {ref}")

for sheet in CSS:
    src = read("assets", sheet)
    for ref in set(re.findall(r"url\(['\"]?(?!data:|https?:)([^'\")]+)", src)):
        if not os.path.exists(os.path.join(WEB, "assets", ref)):
            fail(f"assets/{sheet} references missing file: {ref}")

# --- css custom properties must be defined before they are used ------------
defined = set()
for sheet in CSS:
    defined |= set(re.findall(r"(--[\w-]+)\s*:", read("assets", sheet)))
# Properties the scripts set at runtime are legitimately absent from the
# stylesheets, as are any used with an inline fallback.
from_js = set()
for name in SCRIPTS:
    from_js |= set(re.findall(r"setProperty\(\s*['\"](--[\w-]+)", read("assets", name)))
for sheet in CSS + PAGES:
    src = read("assets", sheet) if sheet in CSS else read(sheet)
    for used, fallback in set(re.findall(r"var\((--[\w-]+)\s*(,?)", src)):
        if used in defined or used in from_js or fallback:
            continue
        fail(f"{sheet} uses undefined custom property {used}")

# --- the tile grid must have room for the tiles it is given ----------------
site = json.load(open(os.path.join(WEB, "data", "site.json")))
n_tiles = len(site["tiles"])
m = re.search(r"\.mosaic\{[^}]*grid-template-columns:repeat\((\d+)", read("assets", "site.css"))
if not m:
    fail("could not read the mosaic column count")
else:
    cols = int(m.group(1))
    if n_tiles % cols:
        fail(f"{n_tiles} tiles in a {cols}-column mosaic leaves "
             f"{cols - n_tiles % cols} empty cells in the last row")

for t in site["tiles"]:
    if not os.path.exists(os.path.join(WEB, "assets", t["file"])):
        fail(f"tile {t['file']} is in site.json but not served")

# --- no page may still point at a renamed script bundle --------------------
for page in PAGES:
    if "onnxruntime-web" in read(page) and "ort.wasm.min.js" not in read(page):
        fail(f"{page} loads a non-wasm onnxruntime bundle")

# --- every element id the scripts reach for should exist somewhere ---------
markup = " ".join(read(p) for p in PAGES)
ids = set(re.findall(r'id="([\w-]+)"', markup))
for name in SCRIPTS:
    src = read("assets", name)
    for wanted in set(re.findall(r"getElementById\(['\"]([\w-]+)['\"]\)", src)):
        if wanted not in ids:
            fail(f"assets/{name} looks up #{wanted}, which no page defines")

print(f"checked {len(PAGES)} pages, {n_tiles} tiles, {len(defined)} tokens")
if fails:
    print("\nFAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("clean")
