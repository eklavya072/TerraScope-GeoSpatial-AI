/* Trade-offs: the quantisation cost, and the constraint explorer.
   Every number is read from data/site.json. */
/* Paint how far each constraint has been pushed onto the track itself. */
function paintRange(el) {
  var min = parseFloat(el.min || '0'), max = parseFloat(el.max || '100');
  var pct = max > min ? ((parseFloat(el.value) - min) / (max - min)) * 100 : 0;
  el.style.setProperty('--fill', pct.toFixed(1) + '%');
}

(function () {
  "use strict";
  TS.init();

  var D = null;
  var SLIDER_STEP = 0.05;   // structural: the slider increment, not a result

  function panel(key, q, loss) {
    var model = key.split('|')[0];
    var verdict = loss
      ? 'Post-training quantisation cannot represent this network. Recovering ' +
        'it needs quantisation-aware training, which a post-training benchmark ' +
        'does not do.'
      : 'The interval spans zero, so on this dataset the quantised graph is ' +
        'not distinguishable from its fp32 parent.';
    var change = (q.delta >= 0 ? '+' : '') + q.delta.toFixed(2) +
                 (q.half_width ? ' ± ' + q.half_width.toFixed(2) : '');
    return '<div class="qpanel' + (loss ? ' loss' : '') + '">' +
      '<h3 class="h3 mono">' + model + '</h3>' +
      '<p class="note" style="margin-top:0">' + verdict + '</p>' +
      '<span class="delta ' + (loss ? 'bad' : 'ok') + '">' + change + ' pp</span>' +
      '<span class="label">Accuracy change, fp32 to int8 static</span>' +
      '<div class="pair">' +
        '<div class="pcol"><div class="stem" style="--v:' + (q.fp32 / 100).toFixed(4) + '"></div>' +
        '<span class="tick">fp32 ' + q.fp32.toFixed(2) + '%</span></div>' +
        '<div class="pcol q"><div class="stem" style="--v:' + (q.int8 / 100).toFixed(4) + '"></div>' +
        '<span class="tick">int8 ' + q.int8.toFixed(2) + '%</span></div>' +
      '</div></div>';
  }

  function quantisation() {
    var statics = Object.keys(D.quantisation).filter(function (k) {
      return k.indexOf('int8_static') !== -1;
    });
    var safest = statics.reduce(function (a, b) {
      return D.quantisation[a].delta > D.quantisation[b].delta ? a : b;
    });
    var worst = statics.reduce(function (a, b) {
      return D.quantisation[a].delta < D.quantisation[b].delta ? a : b;
    });
    document.getElementById('quant').innerHTML =
      panel(safest, D.quantisation[safest], false) +
      panel(worst, D.quantisation[worst], true);

    var rows = Object.keys(D.quantisation).sort().map(function (k) {
      var q = D.quantisation[k], parts = k.split('|');
      var half = q.half_width;
      var indistinguishable = half > 0 && Math.abs(q.delta) < half;
      return '<tr><td class="model">' + parts[0] + '</td><td class="model dim">' +
        parts[1] + '</td>' +
        '<td class="num">' + q.fp32.toFixed(2) + '</td>' +
        '<td class="num">' + q.int8.toFixed(2) + '</td>' +
        '<td class="num">' + (q.delta >= 0 ? '+' : '') + q.delta.toFixed(2) +
        (half ? ' ± ' + half.toFixed(2) : '') + '</td>' +
        '<td class="dim">' + (indistinguishable ? 'indistinguishable from fp32'
                                                : 'real loss') + '</td></tr>';
    }).join('');
    document.getElementById('qtable').innerHTML =
      '<thead><tr><th>Model</th><th>Precision</th><th class="num">fp32 %</th>' +
      '<th class="num">int8 %</th><th class="num">Change (pp)</th>' +
      '<th>Verdict</th></tr></thead><tbody>' + rows + '</tbody>';
  }

  /* ---------------------------------------------------- the explorer ---- */

  function withEnergy() {
    return D.configs.filter(function (c) { return c.energy != null; });
  }

  function paretoFrontier(points) {
    return points.filter(function (p) {
      return !points.some(function (q) {
        return q !== p && q.energy <= p.energy && q.accuracy >= p.accuracy &&
               (q.energy < p.energy || q.accuracy > p.accuracy);
      });
    });
  }

  function draw() {
    var pts = withEnergy();
    var minAcc = parseFloat(document.getElementById('acc').value);
    var maxLat = parseFloat(document.getElementById('lat').value);
    document.getElementById('accv').textContent = minAcc.toFixed(2) + '%';
    document.getElementById('latv').textContent = maxLat.toFixed(2) + ' ms';

    var ok = pts.filter(function (p) { return p.accuracy >= minAcc && p.p95 <= maxLat; });

    var v = document.getElementById('verdict');
    if (!ok.length) {
      v.innerHTML = '<div class="console-bar"><span class="label">No candidate</span></div>' +
        '<div class="console-pad"><p class="note" style="margin:0">Nothing measured ' +
        'meets both constraints. Loosen one. This is a real answer about the ' +
        'measured set, not a failure of the tool.</p></div>';
    } else {
      var best = ok.reduce(function (a, b) { return a.energy <= b.energy ? a : b; });
      v.innerHTML =
        '<div class="console-bar"><span class="label">Lowest energy meeting your constraints</span>' +
        '<span class="label">' + ok.length + ' of ' + pts.length + ' qualify</span></div>' +
        '<div class="console-pad"><h3 class="h3 mark mono">' + best.label + '</h3>' +
        '<p class="note" style="margin-top:.4rem">' + best.accuracy.toFixed(2) +
        '% accuracy · ' + best.energy.toFixed(2) + ' J per thousand inferences · ' +
        best.p95.toFixed(2) + ' ms p95 · ' + best.size_mb.toFixed(1) + ' MB on disk</p></div>';
    }

    /* Scatter, drawn as SVG from the data rather than by a chart library. */
    var W = 720, H = 400, L = 62, R = 24, T = 20, B = 52;
    var xs = pts.map(function (p) { return Math.log10(p.energy); });
    var x0 = Math.floor(Math.min.apply(null, xs)), x1 = Math.ceil(Math.max.apply(null, xs));
    var accs = pts.map(function (p) { return p.accuracy; });
    var y0 = Math.max(0, Math.floor(Math.min.apply(null, accs) / 10) * 10 - 5), y1 = 100;
    function X(e) { return L + (Math.log10(e) - x0) / (x1 - x0) * (W - L - R); }
    function Y(a) { return H - B - (a - y0) / (y1 - y0) * (H - T - B); }

    var front = paretoFrontier(pts).sort(function (a, b) { return a.energy - b.energy; });
    var svg = [];
    svg.push('<line class="axis" x1="' + L + '" y1="' + T + '" x2="' + L + '" y2="' + (H - B) + '"/>');
    svg.push('<line class="axis" x1="' + L + '" y1="' + (H - B) + '" x2="' + (W - R) + '" y2="' + (H - B) + '"/>');
    for (var e = x0; e <= x1; e++) {
      svg.push('<text x="' + X(Math.pow(10, e)).toFixed(1) + '" y="' + (H - B + 16) +
               '" text-anchor="middle">' + Math.pow(10, e) + '</text>');
    }
    [y0, Math.round((y0 + 100) / 2), 100].forEach(function (a) {
      svg.push('<text x="' + (L - 8) + '" y="' + (Y(a) + 3).toFixed(1) +
               '" text-anchor="end">' + a + '</text>');
    });
    svg.push('<text x="' + ((L + W - R) / 2) + '" y="' + (H - 12) +
             '" text-anchor="middle">Energy per 1,000 inferences (J, log scale)</text>');
    svg.push('<text transform="rotate(-90 16 ' + (H / 2) + ')" x="16" y="' + (H / 2) +
             '" text-anchor="middle">Test accuracy (%)</text>');
    svg.push('<polyline class="line" points="' + front.map(function (p) {
      return X(p.energy).toFixed(1) + ',' + Y(p.accuracy).toFixed(1);
    }).join(' ') + '"/>');
    pts.forEach(function (p) {
      var meets = ok.indexOf(p) !== -1;
      var cx = X(p.energy), cy = Y(p.accuracy);
      var text = p.label;
      /* Flip the label to the left near the right edge so it never runs off
         the plot. */
      var flip = cx > 300;
      svg.push(
        '<g class="pt' + (meets ? ' meets' : '') + '">' +
        '<circle class="halo" cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) + '" r="13"/>' +
        '<circle class="dot' + (meets ? ' meets' : '') + '" cx="' + cx.toFixed(1) +
          '" cy="' + cy.toFixed(1) + '" r="6"/>' +
        '<text class="tip" x="' + (flip ? cx - 12 : cx + 12).toFixed(1) + '" y="' +
          (cy - 11).toFixed(1) + '"' + (flip ? ' text-anchor="end"' : '') + '>' +
          text + '</text>' +
        '</g>');
    });
    document.getElementById('chart').innerHTML = svg.join('');
  }

  TS.loadData().then(function (d) {
    D = d;
    quantisation();

    var pts = withEnergy();
    var accs = pts.map(function (p) { return p.accuracy; });
    var lats = pts.map(function (p) { return p.p95; });
    var acc = document.getElementById('acc'), lat = document.getElementById('lat');
    acc.min = Math.min.apply(null, accs); acc.max = Math.max.apply(null, accs);
    acc.step = SLIDER_STEP; acc.value = accs.slice().sort(function (a, b) { return a - b; })[Math.floor(accs.length / 2)];
    lat.min = Math.min.apply(null, lats); lat.max = Math.max.apply(null, lats);
    lat.step = SLIDER_STEP; lat.value = lat.max;
    paintRange(acc); paintRange(lat);

    acc.addEventListener('input', draw);
    lat.addEventListener('input', draw);
    draw();
  }).catch(function (e) {
    document.getElementById('quant').innerHTML =
      '<div class="empty">Could not load the benchmark data: ' +
      ((e && e.message) || e) + '</div>';
  });
})();

/* Keep every slider's track in step with its value, including on load. */
(function () {
  var rs = [].slice.call(document.querySelectorAll('input[type=range]'));
  rs.forEach(paintRange);
  document.addEventListener('input', function (e) {
    if (e.target && e.target.type === 'range') paintRange(e.target);
  });
})();
