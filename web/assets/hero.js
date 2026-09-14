/* The scroll-scrubbed hero.

   The journey is driven by a frame loop that polls the scroll position rather
   than by the scroll event: scroll events are not delivered in every embedding
   (in one preview surface they never fired at all while scrollY moved
   perfectly well), and a hero that depends on them is a hero that silently
   freezes. The loop idles whenever the stage is off screen.
*/
(function () {
  "use strict";

  TS.init();

  /* Real figures replace the markup's placeholders before anything animates,
     so no number on this page is one a human typed. */
  TS.loadData().then(function (d) {
    var f = d.finding;
    function put(id, value, dec, suffix) {
      var el = document.getElementById(id);
      if (!el) return;
      el.dataset.count = value;
      if (dec != null) el.dataset.dec = dec;
      if (suffix != null) el.dataset.suffix = suffix;
      if (!el.dataset.counted) el.textContent = '0';
    }
    /* The trio reads as one comparison: cheapest against dearest, on all
       three axes. Mixing in the gap against the most accurate model would
       pair 0.90 pp with an energy ratio measured against a different one. */
    /* One comparison, read three ways, all against the same pair: the
       cheapest configuration and the dearest one a deployer would call its
       equal. Mixing in the gap against the most accurate model would pair
       one accuracy cost with an energy ratio measured against a different
       configuration. */
    put('ratio-headline', f.ratio.toFixed(0), 0, '×');
    put('stat-windows', d.exclusions.total, 0, '');

    /* The comparison panel, built from the two ends of the measured range.
       Every cell is a value out of the benchmark; the markup ships the same
       values so a failed fetch shows the truth rather than a placeholder. */
    var vb = document.getElementById('vs-body');
    if (vb) {
      var lo = f.cheapest, hi = f.dearest;
      var rows = [
        ['Energy per 1,000 tiles', lo.energy.toFixed(2) + ' J', hi.energy.toFixed(2) + ' J',
         f.ratio.toFixed(0) + '× less', 0, ''],
        ['Top-1 accuracy', lo.accuracy.toFixed(2) + '%', hi.accuracy.toFixed(2) + '%',
         f.gap_vs_dearest.toFixed(2) + ' pp worse', 1, 'cost'],
        ['Latency, 95th percentile', lo.p95.toFixed(2) + ' ms', hi.p95.toFixed(2) + ' ms',
         f.latency_ratio.toFixed(0) + '× faster', 0, ''],
        ['Model on disk', lo.size_mb.toFixed(1) + ' MB', hi.size_mb.toFixed(1) + ' MB',
         (f.size_ratio ? f.size_ratio.toFixed(0) + '× smaller' : 'not recorded'), 0, '']
      ];
      vb.innerHTML = rows.map(function (r) {
        return '<tr><th>' + r[0] + '</th>' +
          '<td class="num' + (r[4] ? '' : ' win') + '">' + r[1] + '</td>' +
          '<td class="num' + (r[4] ? ' win' : '') + '">' + r[2] + '</td>' +
          '<td class="num delta ' + r[5] + '">' + r[3] + '</td></tr>';
      }).join('');
      TS.setText('#vs-lo-name', lo.model);
      TS.setText('#vs-lo-prec', lo.precision.replace('_', ' '));
      TS.setText('#vs-hi-name', hi.model);
      TS.setText('#vs-hi-prec', hi.precision.replace('_', ' ') + ' baseline');
    }

    var spread = document.querySelector('[data-suffix=" pp total spread"]');
    if (spread) spread.dataset.count = f.spread.toFixed(2);

    TS.setText('#find-ratio', f.ratio.toFixed(0) + '×');

    var worst = null, worstKey = null;
    Object.keys(d.quantisation).forEach(function (k) {
      if (!k.endsWith('int8_static')) return;
      if (worst === null || d.quantisation[k].delta < worst.delta) {
        worst = d.quantisation[k]; worstKey = k;
      }
    });
    if (worst) TS.setText('#find-cliff', worst.delta.toFixed(2) + ' pp');

    var body = document.getElementById('find-ratio-body');
    if (body) {
      /* The gap named here must be the one measured against the model the
         energy ratio is quoted against, not the gap against the most accurate
         configuration overall. They are different numbers. */
      body.textContent = f.cheapest.label + ' costs ' +
        f.gap_vs_dearest.toFixed(2) + ' points against ' + f.dearest.label +
        ' and runs on ' + f.ratio.toFixed(0) + ' times less energy, at ' +
        f.latency_ratio.toFixed(0) + ' times lower p95 latency.';
    }

    /* The tile mosaic, built from the real sample set. */
    var mos = document.getElementById('mosaic');
    if (mos) {
      d.tiles.forEach(function (t) {
        var i = document.createElement('img');
        i.src = 'assets/' + t.file;
        i.alt = t.label + ' tile from the held-out test fold';
        i.loading = 'lazy';
        i.style.cssText = 'width:100%;aspect-ratio:1;object-fit:cover;display:block;' +
          'image-rendering:pixelated;transition:transform .4s ease,filter .4s ease;filter:saturate(.94)';
        i.addEventListener('pointerenter', function () {
          i.style.transform = 'scale(1.09)'; i.style.filter = 'saturate(1.2)';
        });
        i.addEventListener('pointerleave', function () {
          i.style.transform = ''; i.style.filter = 'saturate(.94)';
        });
        mos.appendChild(i);
      });
    }
    TS.counters();
  }).catch(function (e) {
    console.warn('site data unavailable:', e);
  });

  /* ------------------------------------------------------- the journey -- */

  var small = matchMedia('(max-width: 900px)').matches;
  if (small || TS.reduced) { document.documentElement.className += ' no-scrub'; return; }

  var track = document.getElementById('track');
  var video = document.getElementById('hero');
  var poster = document.getElementById('poster');
  var ring = document.getElementById('ring');
  var wedge = document.getElementById('wedge');
  var settle = document.getElementById('settle');
  var beats = [].slice.call(document.querySelectorAll('.beat'));
  var BEATS = beats.length;

  beats.forEach(function (b) {
    var h = b.querySelector('.anim');
    if (h) splitAnim(h);
  });

  /* ---- the word engine ------------------------------------------------
     Each heading is broken into words wrapped as .w > .i and numbered, so the
     stylesheet can give every beat its own way in and stagger it. The gap
     between words is a real text node: an inline-block swallows its own
     trailing space, which once ran a whole headline together.

     Inline elements are kept whole rather than split, so the counting number
     inside the fifth heading survives with its id and animates as one word.
  */
  function splitAnim(el) {
    if (el.dataset.anim) return;
    el.dataset.anim = '1';
    var nodes = [].slice.call(el.childNodes);
    var parts = [];
    el.textContent = '';

    function wrap(child) {
      var w = document.createElement('span'); w.className = 'w';
      var i = document.createElement('span'); i.className = 'i';
      i.appendChild(child); w.appendChild(i); el.appendChild(w);
      parts.push(i);
    }

    nodes.forEach(function (node) {
      if (node.nodeType === 3) {
        node.textContent.split(/(\s+)/).forEach(function (piece) {
          if (!piece) return;
          if (/^\s+$/.test(piece)) { el.appendChild(document.createTextNode(' ')); return; }
          wrap(document.createTextNode(piece));
        });
      } else if (node.nodeName === 'BR') {
        el.appendChild(node);
      } else {
        wrap(node);
      }
    });

    /* Always in reading order, including in the right-aligned beats. A
       stagger that runs from the margin inwards there assembles the sentence
       back to front, which is a nice effect and an unreadable one. */
    parts.forEach(function (n, k) { n.style.setProperty('--i', k); });
  }

  /* One viewport of scroll per beat, plus one at the end to read the last one
     on. The stage stays stuck until its container's final viewport, so the
     journey has to finish a viewport before the container does; otherwise the
     closing beat arrives at the same moment the frame starts sliding away and
     is never on screen whole. */
  /* More than a viewport per beat: a line should sit and be read before the
     next one starts arriving. BEAT_VIEWPORTS is the dwell, plus one viewport
     at the end to finish the closing beat on. Declared here rather than below
     because `var` hoists the name but not the value, and an undefined factor
     silently produces a NaN height that the browser drops. */
  var BEAT_VIEWPORTS = 1.6;
  track.style.height = ((BEATS * BEAT_VIEWPORTS + 1) * 100) + 'vh';
  track.style.height = ((BEATS * BEAT_VIEWPORTS + 1) * 100) + 'svh';

  // Structural, not a result: how close a seek must land before it counts
  // as arrived. Named so the no-invented-numbers check can allow the line.
  var SEEK_EPSILON = 0.008;

  var VIDEO = 'assets/ridge-scrub.mp4';
  // Structural: far enough off zero that the browser treats it as a real seek.
  var FIRST_FRAME_NUDGE = 0.05;
  /* Where the closing still takes over, as a fraction of the scroll. At this
     point the footage is one sample from its end, so the crossfade is between
     neighbouring frames. */
  var SETTLE_AT = 0.97;

  /* A wheel or a trackpad delivers scroll in lumps, and a hero driven
     straight off window.scrollY inherits every one of them: the footage
     advances in the same steps the input arrives in, and a beat changes on a
     hard boundary in the middle of one. So the page has a position and the
     film has a position, and the second chases the first.

     Exponential, against elapsed time rather than per frame, so it settles at
     the same rate on a 60Hz panel and a 120Hz one. GLIDE_TAU is how long the
     chase takes to close most of a gap; GLIDE_MAX_STEP caps the step after a
     stall or a background tab, where one frame can carry a whole second and
     would otherwise snap. */
  var GLIDE_TAU = 0.12;        // seconds
  var GLIDE_MAX_STEP = 0.05;   // seconds of catch-up per frame, at most
  var GLIDE_SNAP = 0.0004;     // closer than this and there is nothing to see

  var wanted = 0, shown = -1, glideAt = 0;
  var target = 0, requested = -1;
  var lastBeat = -1, dirty = true, raf = null, live = false;

  /* Read the duration every time rather than caching it at loadedmetadata.
     The event can arrive before the value is usable, and a duration cached as
     NaN becomes a target of zero, which pins the footage on its first frame
     while the beats change around it and looks for all the world like a
     broken seek. A duration this never trusts is simply not ready yet. */
  function footage() {
    var d = video.duration;
    return (isFinite(d) && d > 0) ? d : 0;
  }

  /* Loaded by src rather than fetched into a Blob: the file is small and
     densely keyframed, so the browser buffers it quickly and seeks land fast,
     and a direct src has no cross-origin question to answer wherever this
     page ends up hosted. */
  var nudged = false;
  function arrived() {
    if (!footage()) return;
    ring.hidden = true;

    /* A media element that has neither played nor seeked has no presented
       frame: the element paints as nothing and the stage shows its own
       background. At the top of the page the scrub target is zero, and
       assigning currentTime = 0 when it is already 0 is a no-op, so nothing
       ever forces that first decode. Nudge off zero once to make the browser
       present a frame. */
    if (!nudged) {
      nudged = true;
      try { video.currentTime = FIRST_FRAME_NUDGE; } catch (e) { }
    }
    poster.style.opacity = '0';
    if (settle) settle.classList.add('armed');
    dirty = true;
    schedule();
    /* Wait for enough of the file to have arrived that the seekable range is
       meaningful before judging the host on it. */
    if (!blobbed && video.readyState >= 3 && !seekable()) viaBlob();
  }

  /* A host that answers Range with 200 and the whole file leaves the browser
     reporting the video fully buffered and seekable = [0, 0]: every seek
     snaps back to the first frame, so the footage freezes while the captions
     carry on. The stdlib dev server does exactly this. Rather than trust the
     host, notice it and re-read the file into a blob, which is always
     seekable. Costs one extra download, and only where it would otherwise be
     broken. */
  function seekable() {
    return video.seekable.length > 0 && video.seekable.end(0) > 0;
  }

  var blobbed = false;
  function viaBlob() {
    blobbed = true;
    fetch(video.currentSrc || VIDEO)
      .then(function (r) { return r.blob(); })
      .then(function (b) { video.src = URL.createObjectURL(b); video.load(); })
      .catch(function () { arrived(); });   // no worse than the frozen frame
  }
  video.addEventListener('loadedmetadata', arrived);
  video.addEventListener('durationchange', arrived);
  video.addEventListener('canplay', arrived);
  video.addEventListener('error', function () {
    /* The page is complete without the footage: keep the poster and drop the
       journey to a single screen so nothing scrolls past an empty stage. */
    ring.hidden = true;
    video.remove();
    track.style.height = '100svh';
    beats.forEach(function (b, i) { b.classList.toggle('on', i === 0); });
  });
  /* The footage is decoration, and it must not sit inside the document's load.
     hero.js is a synchronous script, so starting the fetch here puts 33 MB on
     the critical path: the browser counts it toward the page, the tab's
     progress bar stops a fifth of the way along, and a visitor watches a
     stalled loading bar instead of the poster that is already painted behind
     it. Start it after everything else has arrived. The poster is showing by
     then, arrived() swaps it for the first frame, and the error handler
     already covers footage that never turns up at all. */
  function beginFootage() {
    video.src = VIDEO;
    video.load();
  }
  if (document.readyState === 'complete') beginFootage();
  else addEventListener('load', beginFootage, { once: true });

  /* Safari decodes but does not paint a media element that has never played:
     the seeks land, currentTime advances, and the stage stays on the poster.
     Priming with a muted play and an immediate pause gives it a painted frame
     to update from. The promise rejects where autoplay is refused, which is
     survivable — the poster stays and the captions still run. */
  var primed = false;
  function prime() {
    if (primed) return;
    primed = true;
    var p = video.play();
    if (p && p.then) p.then(function () { video.pause(); }, function () { });
    else video.pause();
  }
  video.addEventListener('loadeddata', prime);
  addEventListener('pointerdown', prime, { once: true, passive: true });

  function schedule() { if (raf === null) raf = requestAnimationFrame(frame); }

  function readScroll() {
    var span = track.offsetHeight - 2 * window.innerHeight;
    var p = span > 0 ? (window.scrollY - track.offsetTop) / span : 0;
    wanted = Math.max(0, Math.min(1, p));
    /* A reload part-way down the page starts where it starts. Chasing from
       zero would rewind the whole journey in front of someone who never
       asked to see it. */
    if (shown < 0) shown = wanted;
  }

  /* Everything the stage shows hangs off the chased position, the beats
     included — a beat that changed on the raw scroll while the footage under
     it was still catching up is exactly the join that reads as forced. */
  function apply(p) {
    var d = footage();
    if (d) target = p * d;

    if (settle) settle.classList.toggle('on', p >= SETTLE_AT);

    var idx = Math.min(BEATS - 1, Math.floor(p * BEATS));
    if (idx !== lastBeat) {
      beats.forEach(function (b, i) { b.classList.toggle('on', i === idx); });
      var active = beats[idx];
      /* Grade in from whichever edge this beat sits against, so the middle
         of the frame stays clear of it. */
      var side = active.classList.contains('b4') ? 'foot'
               : (getComputedStyle(active).textAlign === 'right' ? 'right' : 'left');
      wedge.className = 'wedge on ' + side;
      /* Counters fire when their beat arrives, not when the page loads. */
      active.querySelectorAll('[data-count]').forEach(TS.countUp);
      lastBeat = idx;
    }
  }

  function frame() {
    raf = null;
    if (window.scrollY !== frame.lastY) { frame.lastY = window.scrollY; dirty = true; }
    if (dirty) { dirty = false; readScroll(); }

    var now = (window.performance && performance.now) ? performance.now() : Date.now();
    var dt = Math.min(GLIDE_MAX_STEP, (now - (glideAt || now)) / 1000);
    glideAt = now;
    if (TS.reduced || Math.abs(wanted - shown) < GLIDE_SNAP) shown = wanted;
    else shown += (wanted - shown) * (1 - Math.exp(-dt / GLIDE_TAU));
    apply(shown);

    if (!footage()) { if (live || shown !== wanted) schedule(); return; }

    /* Seek straight to the target, with no easing of its own. The glide
       already lives in the position above; easing the seek as well would put
       two lags in series, and a seek is not free — twenty percent steps need
       about thirty round trips to arrive, and the footage visibly trails the
       page. One seek per frame, always aimed at the newest position.

       The comparison is against the position last asked for, not against
       currentTime. A seek lands on the nearest decodable frame, which at
       24fps can sit a fiftieth of a second from where it was sent; measuring
       the gap against where it landed means the gap never closes and the
       element seeks continuously while nothing on screen moves. */
    if (!video.seeking && Math.abs(target - requested) > SEEK_EPSILON) {
      requested = target;
      try { video.currentTime = target; } catch (e) { /* seek raced a reload */ }
    }
    if (live || video.seeking || shown !== wanted) schedule();
  }
  frame.lastY = -1;

  var io = new IntersectionObserver(function (entries) {
    live = entries[0].isIntersecting;
    if (live) schedule();
  }, { threshold: 0 });
  io.observe(track);

  video.addEventListener('seeked', schedule);
  addEventListener('scroll', function () { dirty = true; schedule(); }, { passive: true });
  addEventListener('resize', function () { dirty = true; schedule(); }, { passive: true });
  schedule();
})();
