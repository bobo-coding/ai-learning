"""Fail if the built site contains an internal link that does not resolve.

    python -m scripts.check_site_links site

`mkdocs build --strict` already catches links it knows about -- ones written as
`.md` targets in the source. It does not catch a link that `build_docs.py`
rewrote incorrectly into a path that happens to look absolute, or a nav entry
pointing at a page that was never generated. This walks the actual HTML output
and resolves every relative `href` against the filesystem, which is the only
check that matches what a reader's browser will do.

`404.html` is skipped on purpose: it is served from arbitrary URLs, so MkDocs
writes its asset references root-absolute (`/ai-learning/...`). Those are
correct on the deployed site and unresolvable locally, so checking them here
would only produce false failures.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HREF = re.compile(r'(?:href|src)="([^"]+)"')
EXTERNAL = ("http://", "https://", "//", "mailto:", "tel:", "data:", "#")


def check(site: Path) -> tuple[int, list[str]]:
    """Return (number of internal links checked, list of broken ones)."""
    if not site.is_dir():
        raise SystemExit(f"{site} is not a directory -- run `mkdocs build` first")

    pages = sorted(site.rglob("*.html"))
    if not pages:
        raise SystemExit(f"no HTML found under {site} -- did the build produce anything?")

    checked = 0
    broken: list[str] = []
    for page in pages:
        if page.name == "404.html":
            continue
        for match in HREF.finditer(page.read_text(encoding="utf-8")):
            href = match.group(1)
            if href.startswith(EXTERNAL):
                continue
            target = href.split("#", 1)[0].split("?", 1)[0]
            if not target:
                continue
            checked += 1
            resolved = (page.parent / target).resolve()
            # A directory-style link (`../02_attention/`) is served by its index.
            if resolved.exists() or (resolved / "index.html").exists():
                continue
            broken.append(f"{page.relative_to(site)} -> {href}")
    return checked, broken


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    site = Path(argv[0] if argv else "site")
    checked, broken = check(site)
    n_pages = len([p for p in site.rglob("*.html") if p.name != "404.html"])
    print(f"checked {checked} internal links across {n_pages} pages")
    if broken:
        print(f"{len(broken)} broken:")
        for b in broken:
            print(f"  {b}")
        return 1
    print("no broken internal links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
