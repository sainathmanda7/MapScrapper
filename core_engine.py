"""Core engine: scrape Google Maps photos, pick the best ones, upscale and polish.

Returns exactly what app.py expects:
    {"photos": [{"original": bytes, "enhanced": bytes}, ...], "message": str}
"""

import asyncio
import math
import re
from pathlib import Path
from urllib.parse import quote_plus

import cv2
import numpy as np
import requests
from playwright.async_api import async_playwright

# --- settings ---------------------------------------------------------------
MODEL_PATH = Path("models/FSRCNN_x2.pb")  # optional; falls back to Lanczos x2
UPSCALE_BELOW = 3000       # upscale x2 when the long side is below this (px)
MIN_WIDTH = 800            # discard candidates narrower than this (px)
CANDIDATE_FACTOR = 3       # scrape this many times more photos than requested
MAX_CANDIDATES = 36
DUPLICATE_BITS = 6         # perceptual-hash distance treated as "same photo"
PHOTO_PATHS = ("/p/", "/gps-cs-s/", "/geougc-cs/", "/gps-proxy/")  # excludes avatars (/a/, /a-/)
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Collects image URLs from <img> tags AND background-image styles (gallery tiles).
_EXTRACT_JS = r"""
() => {
  const out = new Set();
  const re = /https:\/\/lh\d\.googleusercontent\.com\/[^"')\s]+/;
  document.querySelectorAll('*').forEach(el => {
    const cands = [el.currentSrc || el.src || '', getComputedStyle(el).backgroundImage || ''];
    for (const s of cands) {
      const m = s.match(re);
      if (m) out.add(m[0]);
    }
  });
  return [...out];
}
"""


def _report(report, message: str) -> None:
    if report:
        report(message)


# --- scraping ---------------------------------------------------------------
async def _accept_cookies(page) -> None:
    button = page.get_by_role("button", name=re.compile(r"accept all|i agree", re.I))
    try:
        await button.first.click(timeout=4000)
        await page.wait_for_load_state("domcontentloaded")
    except Exception:
        pass  # no consent screen


async def _open_first_place(page) -> None:
    """If the search returns a results list, click the first result."""
    try:
        await page.wait_for_selector('div[role="feed"], h1', timeout=15000)
    except Exception:
        return
    first = page.locator('div[role="feed"] a[href*="/maps/place/"]').first
    try:
        if await first.count() > 0:
            await first.click(timeout=5000)
            await page.wait_for_timeout(3000)
    except Exception:
        pass


async def _open_photos(page) -> None:
    options = [
        page.get_by_role("tab", name=re.compile(r"photos", re.I)),
        page.locator('button[aria-label*="Photo" i]'),
    ]
    for locator in options:
        try:
            if await locator.count() > 0:
                await locator.first.click(timeout=5000)
                await page.wait_for_timeout(2500)
                return
        except Exception:
            continue


async def _collect_urls(page, limit: int, report=None) -> list:
    found = {}  # base url -> None (ordered, de-duplicated)
    await page.mouse.move(220, 500)  # hover the left panel so wheel scrolls the gallery
    for _ in range(12):
        for url in await page.evaluate(_EXTRACT_JS):
            base = url.split("=")[0]
            if any(tag in base for tag in PHOTO_PATHS):
                found.setdefault(base, None)
        _report(report, f"Collected {len(found)} candidate photos…")
        if len(found) >= limit:
            break
        await page.mouse.wheel(0, 2500)
        await page.wait_for_timeout(900)
    return list(found)[:limit]


async def get_maps_photos(query: str, limit: int, report=None) -> list:
    """Return base image URLs (no size suffix) for the place matching `query`."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        try:
            context = await browser.new_context(
                locale="en-US", viewport={"width": 1400, "height": 900}, user_agent=UA
            )
            page = await context.new_page()
            _report(report, "Opening Google Maps…")
            await page.goto(
                f"https://www.google.com/maps/search/{quote_plus(query)}?hl=en",
                wait_until="domcontentloaded",
            )
            await _accept_cookies(page)
            await _open_first_place(page)
            _report(report, "Opening the photo gallery…")
            await _open_photos(page)
            return await _collect_urls(page, limit, report)
        finally:
            await browser.close()


# --- downloading ------------------------------------------------------------
def _download(base_url: str):
    """Try the original size first (=s0), then a 2048px version."""
    for suffix in ("=s0", "=s2048"):
        try:
            resp = requests.get(base_url + suffix, headers={"User-Agent": UA}, timeout=25)
            if resp.status_code == 200 and resp.content:
                return resp.content
        except requests.RequestException:
            continue
    return None


def _decode(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# --- ranking ----------------------------------------------------------------
def score_image(img) -> float:
    """Heuristic quality score: sharpness + resolution - bad exposure."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (800, max(1, int(800 * h / w))), interpolation=cv2.INTER_AREA)
    sharpness = cv2.Laplacian(small, cv2.CV_64F).var()
    megapixels = (w * h) / 1e6
    exposure_penalty = abs(float(small.mean()) - 120.0) / 120.0
    return math.log1p(sharpness) + 0.5 * math.log(max(megapixels, 0.1)) - 1.5 * exposure_penalty


def dhash(img) -> np.ndarray:
    gray = cv2.cvtColor(cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return (gray[:, 1:] > gray[:, :-1]).flatten()


def select_best(candidates: list, count: int) -> list:
    """Highest score first, skipping near-duplicates."""
    chosen = []
    for cand in sorted(candidates, key=lambda c: c["score"], reverse=True):
        if all(np.count_nonzero(cand["hash"] != c["hash"]) > DUPLICATE_BITS for c in chosen):
            chosen.append(cand)
        if len(chosen) >= count:
            break
    return chosen


# --- upscale + polish -------------------------------------------------------
_sr = None


def _get_upscaler():
    global _sr
    if _sr is None and MODEL_PATH.exists() and hasattr(cv2, "dnn_superres"):
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(MODEL_PATH))
        sr.setModel("fsrcnn", 2)
        _sr = sr
    return _sr


def upscale(img):
    if max(img.shape[:2]) >= UPSCALE_BELOW:
        return img  # already large; upscaling would add nothing
    sr = _get_upscaler()
    if sr is not None:
        try:
            return sr.upsample(img)
        except cv2.error:
            pass
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)


def polish_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    balanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    h, s, v = cv2.split(cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV))
    s = cv2.add(s, 12)
    vibrant = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    blur = cv2.GaussianBlur(vibrant, (0, 0), 1.5)
    return cv2.addWeighted(vibrant, 1.3, blur, -0.3, 0)


def enhance(data: bytes) -> bytes:
    img = _decode(data)
    out = polish_image(upscale(img))
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


# --- orchestrator -----------------------------------------------------------
async def fetch_and_enhance(query: str, count: int = 4, report=None) -> dict:
    urls = await get_maps_photos(query, min(count * CANDIDATE_FACTOR, MAX_CANDIDATES), report)
    if not urls:
        return {
            "photos": [],
            "message": (
                "No photos were found. Try a more specific name with the city, "
                "or Google Maps may have changed its page layout."
            ),
        }

    _report(report, f"Downloading {len(urls)} candidates…")
    blobs = await asyncio.gather(*(asyncio.to_thread(_download, u) for u in urls))

    _report(report, "Ranking photos by sharpness, resolution and exposure…")
    candidates = []
    for data in blobs:
        if not data:
            continue
        img = _decode(data)
        if img is None or img.shape[1] < MIN_WIDTH:
            continue
        candidates.append({"data": data, "score": score_image(img), "hash": dhash(img)})
        del img  # keep memory low; chosen photos are decoded again below

    chosen = select_best(candidates, count)
    photos = []
    for i, cand in enumerate(chosen, start=1):
        _report(report, f"Upscaling and polishing photo {i} of {len(chosen)}…")
        try:
            enhanced = await asyncio.to_thread(enhance, cand["data"])
        except Exception as err:
            _report(report, f"Skipped photo {i}: {err}")
            continue
        photos.append({"original": cand["data"], "enhanced": enhanced})

    message = ""
    if not photos:
        message = "Photos were found but none were usable (too small or failed to process)."
    elif len(photos) < count:
        message = f"Only {len(photos)} usable photo(s) found for this listing."
    return {"photos": photos, "message": message}


async def fetch_photos(query: str, count: int = 24, report=None) -> dict:
    urls = await get_maps_photos(query, min(count * CANDIDATE_FACTOR, MAX_CANDIDATES), report)
    if not urls:
        return {
            "photos": [],
            "message": (
                "No photos were found. Try a more specific name with the city, "
                "or Google Maps may have changed its page layout."
            ),
        }

    _report(report, f"Downloading {len(urls)} candidates…")
    blobs = await asyncio.gather(*(asyncio.to_thread(_download, u) for u in urls))

    _report(report, "Ranking photos by sharpness, resolution and exposure…")
    candidates = []
    for data in blobs:
        if not data:
            continue
        img = _decode(data)
        if img is None or img.shape[1] < MIN_WIDTH:
            continue
        candidates.append({"data": data, "score": score_image(img), "hash": dhash(img)})
        del img

    chosen = select_best(candidates, count)
    photos = [cand["data"] for cand in chosen]

    message = ""
    if not photos:
        message = "Photos were found but none were usable (too small or failed to process)."
    elif len(photos) < count:
        message = f"Found {len(photos)} usable photo(s) for this listing."
    return {"photos": photos, "message": message}