"""Generate the MkDocs site sources in `docs/` from the repo's own markdown.

    python -m scripts.build_docs

`lessons/*.md` and `README.md` stay the single source of truth -- they are
written to be read on GitHub as plain files, and this script adapts them for the
website instead of duplicating them:

* `README.md` becomes the site index, `results/README.md` becomes `results.md`,
  and each lesson keeps its filename so the cross-links between lessons work
  unchanged (MkDocs rewrites `.md` to `.html` itself).
* Links that point *out* of the published set -- source files, directories, the
  CUDA README -- are rewritten to absolute GitHub URLs, because those files are
  not on the site.
* Inline code spans that name a real tracked file (`` `minigpt/bpe.py` ``)
  become links to GitHub. On the website a reader wants to click through to the
  code; in the plain file, backticks are the right rendering. This is the one
  transformation that makes the web version better than the source rather than
  merely equivalent.

Standard library only, so the Actions job needs nothing but mkdocs-material.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPO_URL = "https://github.com/bobo-coding/ai-learning"
BLOB = f"{REPO_URL}/blob/main/"
TREE = f"{REPO_URL}/tree/main/"

# Pages published to the site, as (source path, destination filename).
PAGES: list[tuple[str, str]] = (
    [("README.md", "index.md")]
    + [(f"lessons/{p.name}", p.name) for p in sorted((ROOT / "lessons").glob("*.md"))]
    + [("results/README.md", "results.md")]
)

# Source -> destination, used to decide whether a link stays internal.
_INTERNAL = {src: dst for src, dst in PAGES}

# Extensions we are willing to turn into a GitHub link.
LINKABLE_SUFFIXES = {
    ".py", ".cu", ".cuh", ".cpp", ".h", ".md", ".json", ".toml",
    ".yml", ".yaml", ".log", ".txt", ".cfg", ".ini",
}

_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
_PATHISH = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def tracked_files() -> set[str]:
    """Every path git knows about, so we never link to an untracked file.

    Without this check a lesson mentioning `out/pretrain/history.json` -- which
    exists locally but is gitignored -- would get a link that 404s on the
    published site.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        # No git available: fall back to what is on disk, minus the obvious
        # generated directories.
        return {
            str(p.relative_to(ROOT))
            for p in ROOT.rglob("*")
            if p.is_file() and not any(
                part in {".git", ".venv", "out", "docs", "site", "__pycache__"}
                for part in p.relative_to(ROOT).parts
            )
        }
    return {line for line in out.splitlines() if line}


class Rewriter:
    def __init__(self, tracked: set[str] | None = None):
        self.tracked = tracked if tracked is not None else tracked_files()
        self.dirs = {str(Path(f).parent) for f in self.tracked}
        # every ancestor directory, so "kernels/" resolves as well as "kernels/metal/"
        for f in list(self.tracked):
            parts = Path(f).parts
            for i in range(1, len(parts)):
                self.dirs.add("/".join(parts[:i]))

    # ------------------------------------------------------------- resolution

    def repo_path(self, source: str, target: str) -> str | None:
        """Resolve a relative link against `source` into a repo-relative path."""
        if target.startswith(("http://", "https://", "mailto:", "#", "/")):
            return None
        base = (ROOT / source).parent
        try:
            resolved = (base / target).resolve()
            return str(resolved.relative_to(ROOT))
        except (ValueError, OSError):
            return None

    def exists(self, rel: str) -> bool:
        return rel in self.tracked or rel.rstrip("/") in self.dirs

    def github_url(self, rel: str) -> str:
        rel = rel.rstrip("/")
        return (TREE if rel in self.dirs and rel not in self.tracked else BLOB) + rel

    # ------------------------------------------------------------------ links

    def rewrite_link(self, source: str, target: str) -> str:
        """Point a markdown link at the site page, or at GitHub if it is not one."""
        anchor = ""
        if "#" in target and not target.startswith("#"):
            target, anchor = target.split("#", 1)
            anchor = "#" + anchor
        rel = self.repo_path(source, target)
        if rel is None:
            return target + anchor
        if rel in _INTERNAL:
            return _INTERNAL[rel] + anchor
        # `results/` (the directory) should land on the results page, not GitHub
        if rel.rstrip("/") == "results":
            return "results.md" + anchor
        if self.exists(rel):
            return self.github_url(rel) + anchor
        raise ValueError(f"{source}: link target {target!r} does not exist in the repo")

    # ------------------------------------------------------------ code spans

    def linkify_code_span(self, content: str) -> str | None:
        """Return a GitHub URL if this code span names a real tracked path."""
        if "/" not in content or not _PATHISH.match(content):
            return None
        rel = content.rstrip("/")
        if rel in self.tracked and Path(rel).suffix in LINKABLE_SUFFIXES:
            return BLOB + rel
        if content.endswith("/") and rel in self.dirs:
            return TREE + rel
        return None

    # ----------------------------------------------------------------- driver

    def rewrite(self, source: str, text: str) -> str:
        out: list[str] = []
        in_fence = False
        for line in text.splitlines(keepends=True):
            if _FENCE.match(line):
                in_fence = not in_fence
                out.append(line)
                continue
            if in_fence:
                out.append(line)
                continue

            # Links first, so a linkified code span is never re-processed.
            line = _LINK.sub(
                lambda m: f"[{m.group(1)}]({self.rewrite_link(source, m.group(2))})", line
            )
            # Then bare code spans -- but not ones already inside a link label.
            line = self._linkify_outside_links(line)
            out.append(line)
        return "".join(out)

    def _linkify_outside_links(self, line: str) -> str:
        spans = [(m.start(), m.end()) for m in _LINK.finditer(line)]

        def inside_link(pos: int) -> bool:
            return any(s <= pos < e for s, e in spans)

        def repl(m: re.Match) -> str:
            if inside_link(m.start()):
                return m.group(0)
            url = self.linkify_code_span(m.group(1))
            return f"[`{m.group(1)}`]({url})" if url else m.group(0)

        return _CODE_SPAN.sub(repl, line)


def build(docs_dir: Path = DOCS, verbose: bool = True) -> list[Path]:
    rw = Rewriter()
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir(parents=True)

    written = []
    for src, dst in PAGES:
        text = (ROOT / src).read_text(encoding="utf-8")
        out = docs_dir / dst
        out.write_text(rw.rewrite(src, text), encoding="utf-8")
        written.append(out)
        if verbose:
            print(f"  {src:22s} -> docs/{dst}")

    # A tiny stylesheet: the lessons are dense with wide tables and code.
    extra = docs_dir / "stylesheets"
    extra.mkdir()
    (extra / "extra.css").write_text(EXTRA_CSS, encoding="utf-8")

    if verbose:
        print(f"wrote {len(written)} pages to {docs_dir}/")
    return written


EXTRA_CSS = """/* The lessons carry wide tables of measured numbers and long code blocks. */
.md-grid { max-width: 62rem; }

.md-typeset table:not([class]) {
  font-size: 0.72rem;
  display: table;
  width: 100%;
}
.md-typeset table:not([class]) td,
.md-typeset table:not([class]) th { padding: 0.4em 0.7em; }

/* Numbers in tables read much better tabular. */
.md-typeset table:not([class]) td { font-variant-numeric: tabular-nums; }

/* The "Read: ... Tests: ..." line under each lesson title. */
.md-typeset h1 + p { color: var(--md-default-fg-color--light); font-size: 0.8rem; }

/* Keep the prev/next footer of each lesson visually separate. */
.md-typeset hr + p:last-child { font-size: 0.85rem; }
"""


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DOCS), help="output directory (default: docs/)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    build(Path(args.out), verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
