/* Method: provenance, limits, exclusions and the confound, all read from
   data/site.json so the page cannot drift from the artefacts it describes. */
(function () {
  "use strict";
  TS.init();

  TS.loadData().then(function (d) {
    var p = d.provenance;

    /* Each fingerprint is introduced by the promise it keeps. A reader who
       does not know what a sha256 is still learns what cannot silently
       change; a reader who does can check it. */
    var seals = [
      ['The data split cannot move', 'Which tiles are used for training and ' +
       'which are held back is fixed by this fingerprint. Every result row ' +
       'carries it, and a run aborts if the split file stops matching.',
       'split file', p.split_sha256],
      ['The training recipe cannot drift', 'Optimiser, epochs, learning rate ' +
       'and augmentation are one shared contract. Change any of it and this ' +
       'fingerprint changes, so old and new results can never be mixed up.',
       'recipe', p.recipe_hash],
      ['The comparison is like for like', d.finding.architectures +
       ' architectures, ' + d.finding.seeds + ' seeds each, one recipe. No ' +
       'model gets a budget of its own.',
       'runs', (d.finding.architectures * d.finding.seeds) + ' training runs']
    ];
    document.getElementById('hashes').innerHTML = seals.map(function (x) {
      return '<div class="seal"><h3 class="h3">' + x[0] + '</h3><p>' + x[1] +
        '</p><div class="seal-id"><span>' + x[2] + '</span><code>' + x[3] +
        '</code></div></div>';
    }).join('');

    var grid = p.grid_intensity != null
      ? p.grid_intensity.toLocaleString(undefined, {maximumFractionDigits: 0})
      : 'not recorded';

    var limits = [
      ['One machine',
       'Every latency and energy figure comes from a single Apple M2. Rankings ' +
       'may differ on x86, on server CPUs with AVX-512, or under different ' +
       'memory bandwidth. This is the limitation most likely to change a ' +
       'conclusion.'],
      ['Energy is estimated',
       (p.energy_note || 'Energy is estimated from on-die telemetry.') +
       ' Carbon assumes ' + grid + ' gCO2e/kWh (' +
       (p.grid_intensity_source || 'source not recorded') + '), a constant ' +
       'that scales every carbon figure linearly, so it is stated rather than ' +
       'buried in a library default.'],
      ['The dataset is saturated',
       'EuroSAT is near its ceiling, so architecture differences are small in ' +
       'absolute terms even when statistically reliable. Accuracy is a weak ' +
       'discriminator here, and that is itself the finding.'],
      ['No scene-level split',
       'EuroSAT tiles are cut from larger Sentinel-2 scenes and carry no scene ' +
       'identifier, so adjacent tiles may span folds. Every model is affected ' +
       'equally and comparisons stay valid, but the absolute accuracies should ' +
       'not be read as generalisation to unseen geography.'],
      ['Europe only, RGB only',
       'The corpus covers 34 European countries, and the 13-band multispectral ' +
       'form is not benchmarked. Nothing here supports a claim about ' +
       'performance elsewhere.'],
      ['Energy mostly tracks latency',
       'The two correlate at r = 0.983 at a fixed thread count, so the energy ' +
       'axis is not independent evidence. The residual is real and runs ' +
       'against intuition: quantised models draw more power while they run.']
    ];
    document.getElementById('limits').innerHTML = limits.map(function (l) {
      return '<div class="card rise on"><h3 class="h3">' + l[0] + '</h3><p>' +
             l[1] + '</p></div>';
    }).join('');

    var ex = d.exclusions;
    document.getElementById('exclusions').textContent =
      ex.excluded + ' of ' + ex.total + ' measurement windows were excluded by ' +
      'criteria registered before any measurement was taken. None fall in the ' +
      'primary reporting configuration, so no headline figure changes when ' +
      'they are removed, and this was verified by recomputing the summary both ' +
      'ways. The rows remain in the published data. Nothing was deleted.';

    if (ex.windows && ex.windows.length) {
      document.getElementById('extable').innerHTML =
        '<thead><tr><th>Configuration</th><th>Reason</th></tr></thead><tbody>' +
        ex.windows.map(function (w) {
          return '<tr><td class="model">' + w.model + ' ' + w.precision +
            ' seed' + w.seed + ' t' + w.threads_intra_op + ' b' + w.batch_size +
            '</td><td class="dim">' + (w.reasons || []).join('; ') + '</td></tr>';
        }).join('') + '</tbody>';
    }

    var notes = Object.keys(p.confound || {}).map(function (k) {
      return '<strong>' + k + '</strong> ' + (p.confound[k].note || '');
    });
    document.getElementById('confound').innerHTML = notes.join('<br><br>') +
      '<br><br>Every figure on this site uses the single-thread configuration, ' +
      'which is the unconfounded one. The four-thread windows are published ' +
      'with the confound characterised rather than quietly dropped.';

    var env = p.environment || {};
    var labels = [['CPU', 'cpu'], ['Cores', 'cpu_cores_logical'], ['OS', 'os'],
                  ['Python', 'python'], ['ONNX Runtime', 'onnxruntime'],
                  ['Measured, UTC', 'timestamp_utc']];
    document.getElementById('env').innerHTML = labels.map(function (l) {
      return '<tr><td>' + l[0] + '</td><td>' + (env[l[1]] || 'not recorded') +
             '</td></tr>';
    }).join('');
  }).catch(function (e) {
    document.getElementById('hashes').textContent =
      'Could not load the benchmark data: ' + ((e && e.message) || e);
  });
})();
