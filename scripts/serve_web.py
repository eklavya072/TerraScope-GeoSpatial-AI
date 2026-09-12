"""Serve web/ for development: no caching, and byte ranges.

Two things the stdlib handler does not do, both of which look like bugs in the
page rather than bugs in the server:

  * It sends Last-Modified and no Cache-Control, so a browser will happily run
    the previous build of hero.js against the current markup.
  * It ignores Range and answers 200 with the whole file. A browser then
    reports the video as fully buffered but with seekable = [0, 0], and every
    seek silently snaps back to zero. The scroll-scrubbed hero freezes on its
    first frame while the captions change around it.

Neither matters in production; a real static host does both. This exists so
that what runs locally is what visitors get.

    python scripts/serve_web.py [port]
"""

import functools
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
RANGE = re.compile(r"bytes=(\d*)-(\d*)")


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        header = self.headers.get("Range")
        m = RANGE.fullmatch(header.strip()) if header else None
        if not m:
            return super().send_head()

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            fh = open(path, "rb")
        except OSError:
            self.send_error(404)
            return None

        size = os.fstat(fh.fileno()).st_size
        first, last = m.group(1), m.group(2)
        if first:
            start = int(first)
            end = int(last) if last else size - 1
        else:                                   # a suffix range: the last N bytes
            start, end = max(0, size - int(last)), size - 1
        end = min(end, size - 1)
        if start > end:
            fh.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        fh.seek(start)
        return _Slice(fh, end - start + 1)


class _Slice:
    """A file object that stops at the end of the requested range."""

    def __init__(self, fh, remaining):
        self.fh, self.remaining = fh, remaining

    def read(self, n=-1):
        if self.remaining <= 0:
            return b""
        if n is None or n < 0:
            n = self.remaining
        chunk = self.fh.read(min(n, self.remaining))
        self.remaining -= len(chunk)
        return chunk

    def close(self):
        self.fh.close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8610
    print(f"serving {ROOT} on http://localhost:{port}")
    # Threading, because HTTP/1.1 keep-alive on a single-threaded server lets
    # one idle browser connection hold up every other request.
    ThreadingHTTPServer(("127.0.0.1", port),
                        functools.partial(Handler, directory=ROOT)).serve_forever()
