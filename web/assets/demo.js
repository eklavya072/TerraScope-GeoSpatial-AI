/* The live classifier.

   Runs the benchmark's own ONNX graphs through onnxruntime-web. Two kinds of
   number appear in the table and they are never mixed: milliseconds are
   measured here, in this browser, through WebAssembly; accuracy, energy and
   carbon are read from data/site.json, which is exported from the committed
   benchmark. Anything the benchmark does not carry prints as "not measured".
*/
/* AnnualCrop -> Annual crop, SeaLake -> Sea lake, HerbaceousVegetation -> ... */
function spacedLabel(name) {
  var out = name.replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase();
  return out.charAt(0).toUpperCase() + out.slice(1);
}

(function () {
  "use strict";

  TS.init();

  // The benchmark's own preprocessing recipe, ImageNet channel statistics.
  // tests/test_site_data.py asserts these still equal bench.config, because
  // drifting from them would mean this page demonstrated a pipeline the
  // reported accuracies do not describe.
  var NORM_MEAN = [0.485, 0.456, 0.406];
  var NORM_STD = [0.229, 0.224, 0.225];
  var SIZE = 64;              // the input size the benchmark trained on
  var WARMUP = 3, RUNS = 15;  // enough for a stable median without a stall

  /* Only these graphs ship to the browser. ResNet-50 in fp32 is 89 MB, which
     is not a thing to hand a visitor, so it is absent here and its benchmark
     figures are shown on the trade-offs page instead. */
  var AVAILABLE = {
    fp32: ['efficientnet_lite0', 'mobilenetv3_small'],
    int8_static: ['efficientnet_lite0', 'mobilenetv3_large',
                  'mobilenetv3_small', 'mobilevit_s', 'resnet50']
  };

  var DATA = null;
  /* Model bytes are cached; sessions are not. Holding five live sessions
     keeps five WebAssembly arenas alive at once — about 38 MB of weights plus
     each runtime's own allocations — which some browsers refuse outright with
     "no available backend found / RangeError: Out of memory". The bytes are
     the expensive thing to fetch; a session over cached bytes is cheap to
     rebuild, so we build one, run it, and release it before the next. */
  var weights = {};
  var selected = new Set();
  var running = false;

  var els = {
    tile: document.getElementById('tile'),
    clearup: document.getElementById('clearup'),
    diag: document.getElementById('diag'),
    precision: document.getElementById('precision'),
    chips: document.getElementById('chips'),
    tilebox: document.getElementById('tilebox'),
    tilenote: document.getElementById('tilenote'),
    results: document.getElementById('results'),
    caption: document.getElementById('caption'),
    status: document.getElementById('status'),
    progress: document.getElementById('progress'),
    rerun: document.getElementById('rerun'),
    live: document.getElementById('live')
  };

  function status(text, busy) {
    els.status.textContent = text;
    els.status.classList.toggle('busy', !!busy);
  }

  function progress(fraction) {
    els.progress.style.transform = 'scaleX(' + fraction.toFixed(4) + ')';
  }

  function fmt(v, dec) {
    return v == null ? 'not measured' : v.toFixed(dec);
  }

  function configFor(model, precision) {
    if (!DATA) return null;
    for (var i = 0; i < DATA.configs.length; i++) {
      var c = DATA.configs[i];
      if (c.model === model && c.precision === precision) return c;
    }
    return null;
  }

  /* --------------------------------------------------------- the input -- */

  /* An uploaded image is held here and takes precedence over the picker.
     It never leaves the page: the file is read into a canvas by the browser
     and the tensor is built locally, exactly as a bundled tile is. */
  var uploaded = null;

  function currentTile() {
    if (uploaded) return uploaded;
    return DATA.tiles[els.tile.selectedIndex] || DATA.tiles[0];
  }

  function drawTile() {
    var t = currentTile();
    var src = t.src || ('assets/' + t.file);
    els.tilebox.innerHTML =
      '<div class="tile' + (t.answers > 1 ? ' contested' : '') +
      '"><img src="' + src + '" alt="' + (t.nice || t.label) +
      '"><div class="tile-meta"><span>' + SIZE + ' \u00d7 ' + SIZE +
      ' px</span><span class="truth">' + (t.label || 'unknown') +
      '</span></div></div>';
    els.tilenote.textContent = t.src
      ? 'Your image, resized to ' + SIZE + ' \u00d7 ' + SIZE +
        ' and normalised exactly as the benchmark does. It stays in this tab.'
      : (t.answers > 1
          ? 'Held-out test tile. The five models return ' + t.answers +
            ' different answers to it, and ' + t.wrong + ' of them are wrong.'
          : 'Held-out test tile. All five models agree on this one.');
  }

  function tensorFromTile(file) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () {
        var c = document.createElement('canvas');
        c.width = SIZE; c.height = SIZE;
        var g = c.getContext('2d', { willReadFrequently: true });
        g.drawImage(img, 0, 0, SIZE, SIZE);
        var px = g.getImageData(0, 0, SIZE, SIZE).data;
        var n = SIZE * SIZE;
        var f = new Float32Array(3 * n);
        /* Exactly the benchmark's preprocessing: ImageNet channel statistics,
           NCHW, float32. Drifting from it here would mean the page was
           demonstrating a different pipeline than the one it reports. */
        for (var i = 0; i < n; i++) {
          for (var ch = 0; ch < 3; ch++) {
            f[ch * n + i] = ((px[i * 4 + ch] / 255) - NORM_MEAN[ch]) / NORM_STD[ch];
          }
        }
        resolve(new ort.Tensor('float32', f, [1, 3, SIZE, SIZE]));
      };
      img.onerror = function () { reject(new Error('image failed to load')); };
      img.crossOrigin = 'anonymous';
      img.src = /^(data:|blob:)/.test(file) ? file : 'assets/' + file;
    });
  }

  /* The runtime's own wording for a failed allocation is "no available
     backend found", which tells a visitor nothing about what to do. */
  function diagnostics() {
    var w = (window.ort && ort.env && ort.env.wasm) || {};
    var bundle = '(none)';
    /* Matches the file name rather than the path: the runtime is served from
       this origin as vendor/ort/ort.wasm.min.js, which contains no
       "onnxruntime" to look for. */
    [].forEach.call(document.scripts, function (t) {
      if (!t.src) return;
      var file = t.src.split('/').pop();
      if (file.indexOf('ort') === 0 || t.src.indexOf('onnxruntime') !== -1) {
        bundle = file;
      }
    });
    return [
      bundle,
      'v' + ((window.ort && ort.env && ort.env.versions && ort.env.versions.web) || '?'),
      'threads ' + (w.numThreads === undefined ? '?' : w.numThreads),
      'simd ' + (w.simd === undefined ? '?' : w.simd),
      'isolated ' + !!self.crossOriginIsolated,
      (navigator.deviceMemory ? navigator.deviceMemory + 'GB' : 'mem ?')
    ].join(' · ');
  }

  function explain(e) {
    var msg = (e && e.message) || String(e);
    if (/out of memory|no available backend/i.test(msg)) {
      return 'not enough memory for this model in this browser — close other ' +
             'tabs and press Re-time, or try a smaller architecture';
    }
    if (/fetch|network|failed to load|404/i.test(msg)) {
      return 'the model file could not be downloaded — check the connection ' +
             'and press Re-time';
    }
    return msg;
  }

  function softmax(logits) {
    var m = -Infinity, i;
    for (i = 0; i < logits.length; i++) if (logits[i] > m) m = logits[i];
    var sum = 0, out = new Array(logits.length);
    for (i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - m); sum += out[i]; }
    for (i = 0; i < out.length; i++) out[i] /= sum;
    return out;
  }

  function bytesFor(model, precision) {
    var key = model + '|' + precision;
    if (weights[key]) return weights[key];
    var url = 'models/' + model + '_seed0_' + precision + '.onnx';
    weights[key] = fetch(url).then(function (r) {
      if (!r.ok) throw new Error(url + ' ' + r.status);
      return r.arrayBuffer();
    }).catch(function (e) { delete weights[key]; throw e; });
    return weights[key];
  }

  /* Options chosen to keep the allocation small rather than to run fast.
     The CPU memory arena pre-allocates a pool sized for the graph and is the
     usual reason a small model still fails with RangeError: Out of memory;
     memory patterning reserves more of the same. Turning both off costs a
     little speed on a model this size and asks the browser for far less. */
  var LEAN = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    enableCpuMemArena: false,
    enableMemPattern: false,
    executionMode: 'sequential'
  };
  var LEANER = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'basic',
    enableCpuMemArena: false,
    enableMemPattern: false,
    executionMode: 'sequential'
  };

  function createSession(buf) {
    return ort.InferenceSession.create(new Uint8Array(buf), LEAN)
      .catch(function (first) {
        /* Optimisation itself allocates. If the first attempt died on memory,
           the second asks for the least the runtime can do. */
        return ort.InferenceSession.create(new Uint8Array(buf), LEANER)
          .catch(function () { throw first; });
      });
  }

  /* One session at a time: build, use, release. */
  function withSession(model, precision, fn) {
    return bytesFor(model, precision).then(function (buf) {
      return createSession(buf);
    }).then(function (s) {
      return Promise.resolve(fn(s)).then(function (out) {
        if (s.release) { try { s.release(); } catch (e) { } }
        return out;
      }, function (err) {
        if (s.release) { try { s.release(); } catch (e) { } }
        throw err;
      });
    });
  }

  /* ------------------------------------------------------------- run ---- */

  function classify(model, precision, tensor) {
    return withSession(model, precision, function (s) {
      var feeds = {};
      feeds[s.inputNames[0]] = tensor;
      var chain = Promise.resolve();
      for (var w = 0; w < WARMUP; w++) chain = chain.then(function () { return s.run(feeds); });
      return chain.then(function () {
        var times = [];
        var seq = Promise.resolve();
        for (var i = 0; i < RUNS; i++) {
          seq = seq.then(function () {
            var t0 = performance.now();
            return s.run(feeds).then(function (r) {
              times.push(performance.now() - t0);
              return r;
            });
          });
        }
        return seq.then(function (r) {
          times.sort(function (a, b) { return a - b; });
          var probs = softmax(Array.from(r[s.outputNames[0]].data));
          var top = 0;
          for (var i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
          return {
            label: DATA.classes[top],
            confidence: probs[top] * 100,
            ms: times[Math.floor(times.length / 2)]
          };
        });
      });
    });
  }

  function render(rows, precision, truth) {
    if (!rows.length) {
      els.results.innerHTML = '<div class="empty">Select at least one model.</div>';
      els.caption.textContent = '';
      return;
    }
    var body = rows.map(function (r) {
      if (r.error) {
        return '<tr><td class="model">' + r.model + '</td>' +
               '<td colspan="6" class="dim">did not run: ' + r.error + '</td></tr>';
      }
      var verdict = truth == null
        ? '<span class="verdict none">no label</span>'
        : (r.label === truth ? '<span class="verdict hit">correct</span>'
                             : '<span class="verdict miss">wrong</span>');
      var c = r.config;
      return '<tr>' +
        '<td class="model">' + r.model + '</td>' +
        '<td>' + (r.label || '—') + '<br>' + verdict + '</td>' +
        '<td class="num">' + r.confidence.toFixed(1) + '%</td>' +
        '<td class="num">' + r.ms.toFixed(2) + '</td>' +
        '<td class="num dim">' + (c ? c.accuracy.toFixed(2) : 'not measured') + '</td>' +
        '<td class="num dim">' + (c && c.energy != null ? c.energy.toFixed(2) : 'not measured') + '</td>' +
        '<td class="num dim">' + (c ? c.p95.toFixed(2) : 'not measured') + '</td>' +
        '</tr>';
    }).join('');

    els.results.innerHTML =
      '<div class="tscroll"><table class="data"><thead><tr>' +
      '<th>Model</th><th>Predicted</th>' +
      '<th class="num">Conf</th>' +
      '<th class="num">Browser ms</th>' +
      '<th class="num">Acc %</th>' +
      '<th class="num">J/1k</th>' +
      '<th class="num">M2 p95 ms</th>' +
      '</tr></thead><tbody>' + body + '</tbody></table></div>';

    var missed = rows.filter(function (r) { return !r.error && truth && r.label !== truth; });
    var note = 'Browser ms is the median of ' + RUNS + ' runs on your machine ' +
      'through WebAssembly, after ' + WARMUP + ' warm-up runs. Acc %, J/1k and ' +
      'M2 p95 come from the benchmark on an Apple M2 at one thread, not from ' +
      'this machine.';
    if (missed.length) {
      note += ' ' + missed.length + ' of ' + rows.length + ' models read this ' +
        'tile as something else, confidently, and at full speed: ' +
        missed.map(function (r) { return r.model; }).join(', ') + '.';
    }
    els.caption.textContent = note;
    if (els.diag) els.diag.textContent = 'Runtime: ' + diagnostics();
  }

  /* Copy kept beside the markup's version in demo.html: whichever of the two
     runs, the visitor reads the same sentence. */
  var NO_RUNTIME =
    'The inference runtime did not load, so nothing can be timed on this ' +
    'page. Every measured figure in the tables and on the other pages is ' +
    'unaffected \u2014 those come from the committed benchmark, not from ' +
    'your browser.';

  function run() {
    if (running || !DATA) return;

    /* Without the runtime there is nothing to time. Say so once and stop,
       rather than starting a load that can only end in a per-model error and
       leaving the status stuck on "Loading". */
    if (!window.ort || !ort.InferenceSession) {
      els.results.innerHTML = '<div class="empty">' + NO_RUNTIME + '</div>';
      els.rerun.disabled = true;
      progress(0);
      status('Unavailable');
      return;
    }
    var precision = els.precision.value;
    var models = AVAILABLE[precision].filter(function (m) { return selected.has(m); });
    if (!models.length) { render([], precision, null); progress(0); status('Idle'); return; }

    running = true;
    els.rerun.disabled = true;
    var tile = currentTile();
    var done = 0;
    progress(0);
    status('Loading ' + models.length + ' model' + (models.length > 1 ? 's' : ''), true);

    tensorFromTile(tile.src || tile.file).then(function (tensor) {
      var rows = [];
      return models.reduce(function (chain, model) {
        return chain.then(function () {
          status('Timing ' + model, true);
          return classify(model, precision, tensor).then(function (r) {
            rows.push({ model: model, label: r.label, confidence: r.confidence,
                        ms: r.ms, config: configFor(model, precision) });
          }).catch(function (e) {
            rows.push({ model: model, error: explain(e) });
          }).then(function () {
            done++; progress(done / models.length);
            render(rows.slice(), precision, tile.label);
          });
        });
      }, Promise.resolve()).then(function () {
        render(rows, precision, tile.label);
      });
    }).catch(function (e) {
      els.results.innerHTML = '<div class="empty">Could not run: ' +
        ((e && e.message) || e) + '</div>';
    }).then(function () {
      running = false;
      els.rerun.disabled = false;
      status('Done');
      setTimeout(function () { progress(0); }, 600);
    });
  }

  /* ----------------------------------------------------------- chips ---- */

  function drawChips() {
    var precision = els.precision.value;
    var all = AVAILABLE.int8_static;   // the full roster; fp32 ships fewer
    els.chips.innerHTML = '';
    all.forEach(function (m) {
      var runnable = AVAILABLE[precision].indexOf(m) !== -1;
      var on = selected.has(m) && runnable;
      var cfg = configFor(m, precision);
      /* The weight size is the measured figure from the benchmark, shown on
         the chip because clicking it is what costs the visitor the download.
         Anyone on a metered connection can see the price before paying it. */
      var mb = (cfg && cfg.size_mb) ? cfg.size_mb.toFixed(1) + ' MB' : '';
      var b = document.createElement('button');
      b.className = 'chip' + (on ? ' on' : '');
      b.type = 'button';
      b.disabled = !runnable;
      b.title = runnable
        ? (on ? 'Click to remove' : 'Click to add' + (mb ? ' \u2014 downloads ' + mb : ''))
        : m + ' in ' + precision + ' is too large to ship to a browser';
      b.innerHTML = m
        + (on ? ' <span class="x">×</span>'
              : (mb && runnable ? ' <span class="mb">' + mb + '</span>' : ''));
      b.addEventListener('click', function () {
        if (selected.has(m)) selected.delete(m); else selected.add(m);
        drawChips();
        run();
      });
      els.chips.appendChild(b);
    });
  }

  /* ------------------------------------------------------------ boot ---- */

  TS.loadData().then(function (d) {
    DATA = d;

    /* Tiles are named the way a person would refer to them. The filename is
       still the identity in the data; it just is not what the reader has to
       read out of a dropdown. */
    var seenClass = {};
    d.tiles.forEach(function (t, i) {
      seenClass[t.label] = (seenClass[t.label] || 0) + 1;
      t.nice = spacedLabel(t.label) + ' ' + seenClass[t.label];
      var o = document.createElement('option');
      o.textContent = t.nice;
      o.value = String(i);
      els.tile.appendChild(o);
    });

    /* One model on arrival, not five.
       Selecting every model meant a visitor downloaded 38.4 MB of weights
       before touching anything, which is a real cost on the connections this
       project is aimed at and an awkward look for a benchmark about compute
       efficiency. The remaining models are fetched when their chip is
       selected, and bytesFor() caches each one, so adding a model costs only
       that model. The default is the configuration the benchmark recommends;
       every other is one click away. */
    var preferred = (d.finding && d.finding.cheapest && d.finding.cheapest.model) || '';
    if (AVAILABLE.int8_static.indexOf(preferred) !== -1) {
      selected.add(preferred);
    } else if (AVAILABLE.int8_static.length) {
      selected.add(AVAILABLE.int8_static[0]);
    }

    drawTile();
    drawChips();

    els.tile.addEventListener('change', function () {
      uploaded = null;
      if (els.clearup) els.clearup.hidden = true;
      drawTile(); run();
    });

    var up = document.getElementById('upload');
    if (up) {
      up.addEventListener('change', function () {
        var f = up.files && up.files[0];
        if (!f) return;
        uploaded = { src: URL.createObjectURL(f), label: 'your image',
                     nice: f.name, file: f.name };
        if (els.clearup) els.clearup.hidden = false;
        drawTile(); run();
      });
    }
    if (els.clearup) {
      els.clearup.addEventListener('click', function () {
        uploaded = null; els.clearup.hidden = true;
        if (up) up.value = '';
        drawTile(); run();
      });
    }
    els.precision.addEventListener('change', function () { drawChips(); run(); });
    els.rerun.addEventListener('click', function () { run(); });

    run();   // results are on screen when the visitor arrives
  }).catch(function (e) {
    els.results.innerHTML = '<div class="empty">Could not load the benchmark ' +
      'data: ' + ((e && e.message) || e) + '</div>';
    els.live.textContent = 'data unavailable';
  });
})();
