#!/usr/bin/env python3
"""Fetch a public-domain training-image corpus for `plakat bookart` origin-LoRA
training (RFC BOOKART-1 / ROADMAP G0.3).

Downloads illustrations for three public-domain traditions straight from the
Wikimedia Commons MediaWiki API into ``datasets/bookart_training/<artist>/``:

  * beardsley (english)  — Aubrey Beardsley  (d. 1898) — B/W line, ideal
  * hokusai   (japanese) — Katsushika Hokusai (d. 1849) — B/W sumi sketches
  * bilibin   (russian)  — Ivan Bilibin       (d. 1942) — mixed (color desaturated at train time)

All works used are pre-1929 and public domain in the US.

Pure Python 3 stdlib only (urllib.request / json). No third-party deps.

Wikimedia requires a descriptive User-Agent or it answers 403, so every request
carries one. The script is idempotent: already-downloaded files are skipped.

Usage:
    python3 tools/bookart/fetch_training_corpus.py

Re-run any time; it only fetches what is missing.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "plakat-bookart-corpus (vulogov@gmail.com)"
API = "https://commons.wikimedia.org/w/api.php"
MIN_DIM = 400          # skip files whose reported width or height is below this
TARGET_PER_ARTIST = 40  # stop once we have this many good files
GCM_LIMIT = 60

# repo-root-relative output dir (script lives in tools/bookart/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_ROOT = os.path.join(REPO_ROOT, "datasets", "bookart_training")

# (artist_key, [primary_category, fallback_category, ...])
# Fallbacks are only consulted if the primary underperforms the target.
ARTISTS = [
    ("beardsley", [
        "Illustrations by Aubrey Beardsley",
        "Aubrey Beardsley",
    ]),
    # NB: the capitalised "Hokusai Manga" category is an empty container. The
    # real B/W sumi-sketch plates live under the lowercase per-volume categories
    # "Hokusai manga volNN". "100 Views of Mount Fuji" (Fugaku Hyakkei) is also
    # B/W line work and rounds out the target.
    ("hokusai", [
        "Hokusai manga vol01",
        "Hokusai manga vol03",
        "Hokusai manga vol02",
        "100 Views of Mount Fuji",
    ]),
    # NB: "Illustrations by Ivan Bilibin" does not exist on Commons, and the
    # broad "Ivan Bilibin" category is polluted with photos of his grave,
    # portraits of him, and militia-uniform designs. The clean fairy-tale
    # plates live under "Book illustrations by Ivan Bilibin"; postcards and
    # magazine plates round out the target. These skew colour — desaturate /
    # binarise before training (see README).
    ("bilibin", [
        "Book illustrations by Ivan Bilibin",
        "Postcards by Ivan Bilibin",
        "Magazine illustrations by Ivan Bilibin",
    ]),
]

ALLOWED_MIME = {"image/jpeg": "jpg", "image/png": "png"}


def _urlopen_retry(url, timeout, tries=6):
    """GET a URL with the required User-Agent, retrying with exponential backoff
    on HTTP 429 (Wikimedia rate limit) and transient network errors."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = 2.0
    last = None
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:  # noqa: PERF203
            last = e
            if e.code == 429 or 500 <= e.code < 600:
                wait = delay
                # honour Retry-After when the server sends one
                ra = e.headers.get("Retry-After") if e.headers else None
                if ra and ra.isdigit():
                    wait = max(wait, float(ra))
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise last if last is not None else RuntimeError("urlopen failed: " + url)


def _get(params):
    """Issue a GET to the Commons API with the required User-Agent, return JSON."""
    url = API + "?" + urllib.parse.urlencode(params)
    return json.loads(_urlopen_retry(url, timeout=60).decode("utf-8"))


def list_category_files(category):
    """Yield imageinfo dicts for every file member of a Commons category,
    following continue tokens."""
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": "Category:" + category,
        "gcmtype": "file",
        "gcmlimit": str(GCM_LIMIT),
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "format": "json",
    }
    while True:
        data = _get(params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            infos = page.get("imageinfo", [])
            if infos:
                info = dict(infos[0])
                info["_title"] = title
                yield info
        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)
        time.sleep(0.2)  # be polite


def download(url, dest):
    data = _urlopen_retry(url, timeout=120)
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dest)
    return len(data)


def fetch_artist(artist, categories):
    out_dir = os.path.join(OUT_ROOT, artist)
    os.makedirs(out_dir, exist_ok=True)

    # Count what's already on disk (idempotency) toward the target.
    existing = [f for f in os.listdir(out_dir)
                if f.startswith(artist + "_") and not f.endswith(".part")]
    kept = len(existing)
    idx = kept  # next file number
    seen_urls = set()

    print("[%s] starting (already have %d)" % (artist, kept))

    for category in categories:
        if kept >= TARGET_PER_ARTIST:
            break
        if category != categories[0]:
            print("[%s]   primary underperformed; trying fallback '%s'"
                  % (artist, category))
        try:
            members = list(list_category_files(category))
        except Exception as e:  # noqa: BLE001
            print("[%s]   ! category '%s' failed: %s" % (artist, category, e))
            continue

        for info in members:
            if kept >= TARGET_PER_ARTIST:
                break
            mime = info.get("mime", "")
            ext = ALLOWED_MIME.get(mime)
            if ext is None:
                continue  # skip svg / tif / pdf / djvu
            w = info.get("width", 0) or 0
            h = info.get("height", 0) or 0
            if w < MIN_DIM or h < MIN_DIM:
                continue
            url = info.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            dest = os.path.join(out_dir, "%s_%02d.%s" % (artist, idx, ext))
            # idempotent skip: if a file with this number+ext already exists,
            # bump the index until we find a free slot (keeps names stable-ish).
            while os.path.exists(dest):
                idx += 1
                dest = os.path.join(out_dir, "%s_%02d.%s" % (artist, idx, ext))

            try:
                n = download(url, dest)
            except Exception as e:  # noqa: BLE001
                print("[%s]   ! failed %s: %s" % (artist, info.get("_title"), e))
                continue
            kept += 1
            idx += 1
            print("[%s]   + %s  (%dx%d, %d KB)  <- %s"
                  % (artist, os.path.basename(dest), w, h, n // 1024,
                     info.get("_title", "")))
            time.sleep(1.0)  # gentle pacing to stay under the Wikimedia rate limit

    print("[%s] DONE: %d files in %s" % (artist, kept, out_dir))
    return kept


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    totals = {}
    for artist, categories in ARTISTS:
        totals[artist] = fetch_artist(artist, categories)
    print("\n==== corpus summary ====")
    for artist, _ in ARTISTS:
        print("  %-10s %d images" % (artist, totals[artist]))
    grand = sum(totals.values())
    print("  %-10s %d images" % ("TOTAL", grand))
    return 0 if grand > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
