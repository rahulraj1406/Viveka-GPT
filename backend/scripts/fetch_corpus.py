"""
Fetch Complete Works of Swami Vivekananda from Wikisource.
Saves raw HTML per chapter into backend/data/raw/
"""
import re
import httpx
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
import time

BASE = "https://en.wikisource.org"
INDEX = "/wiki/The_Complete_Works_of_Swami_Vivekananda"
PREFIX = "/wiki/The_Complete_Works_of_Swami_Vivekananda/"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; VivekaGPT-corpus-fetcher/1.0; educational use)"
}

_client = httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30)


def _get_soup(path: str) -> BeautifulSoup:
    r = _client.get(BASE + path)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def _get_rest_soup(path: str) -> BeautifulSoup:
    """Fetch via Wikisource REST API which returns fully rendered HTML."""
    from urllib.parse import quote, unquote
    # Decode any existing percent-encoding before re-encoding for the API
    title = unquote(path.lstrip("/wiki/"))
    url = f"https://en.wikisource.org/api/rest_v1/page/html/{quote(title, safe='')}"
    for attempt in range(5):
        r = _client.get(url)
        if r.status_code == 429:
            wait = 2 ** attempt * 5
            time.sleep(wait)
            continue
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    r.raise_for_status()  # re-raise after all retries exhausted


def get_volume_links() -> list[tuple[int, str]]:
    """Return list of (volume_number, section_url) tuples from the index page."""
    soup = _get_soup(INDEX)
    content = soup.find("div", class_="mw-parser-output")
    seen = set()
    results = []
    for a in content.find_all("a", href=True):
        href = a["href"]
        if not href.startswith(PREFIX):
            continue
        # Only depth-2 paths: /Volume_N/SectionName (not deeper)
        tail = href[len(PREFIX):]
        if tail.count("/") != 1:
            continue
        if href in seen:
            continue
        seen.add(href)
        m = re.search(r"/Volume_(\d+)/", href)
        vol_num = int(m.group(1)) if m else 0
        results.append((vol_num, href))
    return results


def get_chapter_links(section_url: str) -> list[tuple[str, str]]:
    """Return list of (chapter_title, chapter_url) for all chapters in a section.

    If the section page has no sub-chapter links it is itself a leaf chapter.
    """
    soup = _get_soup(section_url)
    content = soup.find("div", class_="mw-parser-output")
    if content is None:
        return []

    # Sub-chapters are links one level deeper than the section URL
    seen = set()
    results = []
    for a in content.find_all("a", href=True):
        href = a["href"]
        if not href.startswith(section_url + "/"):
            continue
        # Only direct children (one extra path component)
        tail = href[len(section_url) + 1:]
        if "/" in tail:
            continue
        title = a.get_text().strip()
        if not title or href in seen:
            continue
        seen.add(href)
        results.append((title, href))

    if not results:
        # Section page is itself the leaf chapter
        title = section_url.rsplit("/", 1)[-1].replace("_", " ")
        return [(title, section_url)]

    return results


def fetch_chapter(url: str) -> str:
    """Return cleaned HTML for one chapter (paragraph tags only)."""
    soup = _get_rest_soup(url)

    # Strip navigation headers, noprint UI chrome, style blocks, footnote refs
    for sel in [
        ".ws-header", ".ws-noexport", ".noprint", ".dynlayout-exempt",
        "style", "sup.reference", ".reflist", ".references", ".mw-editsection",
    ]:
        for tag in soup.select(sel):
            tag.decompose()

    # Collect non-empty paragraphs
    paras = [str(p) for p in soup.find_all("p") if p.get_text().strip()]
    return "\n".join(paras)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after downloading this many new files (0 = no limit)")
    parser.add_argument("--volumes", type=str, default="",
                        help="Comma-separated list of volumes to fetch, e.g. 3,4,5 (default: all)")
    args = parser.parse_args()

    only_vols = set(int(v) for v in args.volumes.split(",") if v.strip()) if args.volumes else set()

    sections = get_volume_links()
    print(f"Found {len(sections)} sections across volumes")

    downloaded = 0
    for vol_num, section_url in tqdm(sections, desc="Sections"):
        if only_vols and vol_num not in only_vols:
            continue
        if args.limit and downloaded >= args.limit:
            break
        chapters = get_chapter_links(section_url)
        for title, url in tqdm(chapters, desc=f"Vol {vol_num}", leave=False):
            if args.limit and downloaded >= args.limit:
                break
            # Sanitize filename: keep alphanumeric, hyphens, underscores
            slug = re.sub(r"[^\w\-]", "_", title.lower())[:80]
            path = RAW_DIR / f"vol{vol_num}_{slug}.html"
            if path.exists():
                continue
            time.sleep(1.0)  # be polite to Wikisource (runs before every fetch)
            html = fetch_chapter(url)
            if html:
                path.write_text(html, encoding="utf-8")
                downloaded += 1

    print(f"\nDone. {downloaded} new files written to {RAW_DIR}")
