/* Shared behaviour: nav spotlight, scroll reveals, counters, and the data
   load. Every figure the site prints comes from data/site.json, which
   scripts/build_site_data.py exports from the committed benchmark. Nothing
   here invents a number. */
(function (global) {
  "use strict";

  var reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  /* Structural: how much of a block must be showing before it starts to
     arrive. It used to be a third, which meant the block was already well up
     the screen before it began — you watched it happen. Low enough now that
     it starts as it clears the fold and has settled by the time it is in
     front of you. */
  var REVEAL_THRESHOLD = 0.12;

  /* ---- the instrument bar -------------------------------------------- */
  function nav() {
    var bar = document.querySelector('.nav');
    if (!bar) return;
    var links = document.getElementById('navlinks');
    var ind = document.getElementById('navpill');
    var prog = document.getElementById('navprog');
    var wide = !matchMedia('(max-width: 900px)').matches;

    /* The rule slides to whichever stop you are on, or pointing at. */
    /* The pill parks on the current page and follows the pointer from
       there, so it never has to appear out of nowhere. */
    function moveTo(a) {
      if (!ind || !a) return;
      ind.style.setProperty('--ix', a.offsetLeft + 'px');
      ind.style.setProperty('--iw', a.offsetWidth + 'px');
    }
    function home() { moveTo(links && links.querySelector('a.here')); }

    if (links && ind) {
      links.querySelectorAll('a').forEach(function (a) {
        a.addEventListener('pointerenter', function () { moveTo(a); });
      });
      links.addEventListener('pointerleave', home);
      addEventListener('resize', home, { passive: true });
      /* Fonts land after first paint and change every offset under it. */
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(home);
      requestAnimationFrame(home);
    }

    /* The bar only takes glass once the hero has finished. While the film is
       still on screen it stays fully transparent, because the whole point of
       a scroll-scrubbed hero is that nothing sits on top of it. The threshold
       is the point where the sticky stage releases, not an arbitrary 40px. */
    function heroEnd() {
      var t = document.getElementById('track');
      if (t && t.offsetHeight) {
        return t.offsetTop + t.offsetHeight - window.innerHeight;
      }
      var h = document.querySelector('.static-hero');
      if (h && h.offsetHeight) return h.offsetTop + h.offsetHeight - bar.offsetHeight;
      return 40;
    }

    /* Scroll: condense the bar, and draw how far down the page you are. */
    var pending = false;
    function onScroll() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () {
        pending = false;
        var y = window.scrollY || 0;
        bar.classList.toggle('tight', y > heroEnd());
        if (prog) {
          var span = document.documentElement.scrollHeight - window.innerHeight;
          prog.style.setProperty('--sp', span > 0 ? (y / span).toFixed(4) : '0');
        }
      });
    }
    addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    if (wide) {
      bar.addEventListener('pointermove', function (e) {
        var r = bar.getBoundingClientRect();
        bar.style.setProperty('--mx', (e.clientX - r.left) + 'px');
        bar.style.setProperty('--my', (e.clientY - r.top) + 'px');
      });
      bar.addEventListener('pointerleave', function () {
        bar.style.setProperty('--mx', '-500px');
      });
    }
  }

  /* The bar carries the headline figure on every page. It is filled from the
     benchmark like everything else; the markup ships an em dash, never a
     number, so a failed fetch cannot leave a stale one on screen. */
  function navFigure() {
    var el = document.getElementById('nav-fig');
    if (!el) return;
    loadData().then(function (d) {
      el.textContent = d.finding.ratio.toFixed(0) + '×';
    }).catch(function () { });
  }

  /* Split into words once, so the reveal can stagger them. The gap is a real
     text node: an inline-block swallows its own trailing space. */
  function splitWords(el) {
    if (el.dataset.split) return;
    el.dataset.split = '1';
    var words = el.textContent.trim().split(/\s+/);
    el.textContent = '';
    words.forEach(function (w, i) {
      var s = document.createElement('span');
      s.textContent = w;
      s.style.transitionDelay = (i * 26) + 'ms';
      el.appendChild(s);
      if (i < words.length - 1) el.appendChild(document.createTextNode(' '));
    });
  }

  function reveals() {
    document.querySelectorAll('.reveal').forEach(splitWords);
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add('on'); io.unobserve(e.target); }
      });
    }, { threshold: REVEAL_THRESHOLD });
    document.querySelectorAll('.reveal, .rise').forEach(function (el) { io.observe(el); });
  }

  /* A line that resolves against scroll position rather than firing once on
     entry. Each word carries its own point in the sweep, so the sentence
     reads as it arrives and reverses if you scroll back — a one-shot fade
     cannot do either, and repeating that same fade on every block is the
     tell of a page with no authored moment in it. */
  function scrubReveal() {
    var els = [].slice.call(document.querySelectorAll('.scrub'));
    if (!els.length) return;
    els.forEach(splitWords);
    if (reduced) return;

    /* Structural, not results: where in the viewport the sweep starts and
       finishes, how dim a word rests before its turn, and how close to done
       counts as done. Named so the no-invented-numbers check can tell them
       apart from a measurement. */
    var SCRUB_ENTER = 0.88;      // fraction of viewport height: band top
    var SCRUB_EXIT = 0.36;       // fraction of viewport height: band bottom
    var SCRUB_FLOOR = 0.16;      // resting opacity of a word not yet reached
    var SCRUB_DONE = 0.98;       // past this, drop the blur entirely

    var raf = null;
    function paint() {
      raf = null;
      els.forEach(function (el) {
        var top = el.getBoundingClientRect().top;
        var from = window.innerHeight * SCRUB_ENTER;
        var to = window.innerHeight * SCRUB_EXIT;
        var p = Math.max(0, Math.min(1, (from - top) / (from - to)));
        var words = el.querySelectorAll('span');
        var n = words.length;
        for (var i = 0; i < n; i++) {
          /* The sweep runs a little past the last word so the tail finishes
             before the line leaves the band. */
          var w = Math.max(0, Math.min(1, (p * (n + 3) - i) / 2.5));
          words[i].style.opacity = (SCRUB_FLOOR + (1 - SCRUB_FLOOR) * w).toFixed(3);
          words[i].style.filter = w > SCRUB_DONE
            ? 'none' : 'blur(' + ((1 - w) * 4).toFixed(2) + 'px)';
        }
      });
    }
    function schedule() { if (raf === null) raf = requestAnimationFrame(paint); }
    addEventListener('scroll', schedule, { passive: true });
    addEventListener('resize', schedule, { passive: true });
    paint();
  }

  function format(v, dec, suffix) {
    return v.toFixed(dec) + (suffix || '');
  }

  /* Count from zero to the element's real value. */
  function countUp(el) {
    if (el.dataset.counted) return;
    el.dataset.counted = '1';
    var target = parseFloat(el.dataset.count);
    var dec = parseInt(el.dataset.dec || '0', 10);
    var suffix = el.dataset.suffix || '';
    if (reduced || !isFinite(target)) { el.textContent = format(target, dec, suffix); return; }
    var t0 = null, dur = 1150;
    requestAnimationFrame(function step(ts) {
      if (t0 === null) t0 = ts;
      var p = Math.min((ts - t0) / dur, 1);
      el.textContent = format(target * (1 - Math.pow(1 - p, 3)), dec, suffix);
      if (p < 1) requestAnimationFrame(step);
    });
  }

  function counters(scope) {
    /* Figures inside a hero beat are driven by that beat, not by visibility:
       they sit in the viewport from the first paint even while transparent,
       so a plain observer would count them down to nothing before anyone
       could see them. */
    var els = (scope || document).querySelectorAll('[data-count]:not(.beat [data-count])');
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) { countUp(e.target); io.unobserve(e.target); }
      });
    }, { threshold: 0.6 });
    els.forEach(function (el) { io.observe(el); });
  }

  function setText(sel, value) {
    document.querySelectorAll(sel).forEach(function (el) { el.textContent = value; });
  }

  function markNav() {
    var here = location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav-links a').forEach(function (a) {
      var target = a.getAttribute('href');
      if (target === here) a.classList.add('here');
    });
  }

  function loadData() {
    return fetch('data/site.json').then(function (r) {
      if (!r.ok) throw new Error('site.json ' + r.status);
      return r.json();
    });
  }

  global.TS = {
    reduced: reduced,
    init: function () { markNav(); nav(); navFigure(); reveals(); scrubReveal(); counters(); },
    counters: counters,
    countUp: countUp,
    setText: setText,
    loadData: loadData,
    splitWords: splitWords
  };
})(window);
