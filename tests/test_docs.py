"""Tests for the site generator.

The lessons are the single source of truth and `build_docs.py` rewrites their
links for the web. A silent mistake there ships 404s to readers, so the
rewriting rules are tested directly rather than only exercised by a CI build.
"""

import re

import pytest

from scripts.build_docs import PAGES, ROOT, Rewriter, build
from scripts.check_site_links import check

TRACKED = {
    "README.md",
    "lessons/00_setup.md",
    "lessons/01_tokenization.md",
    "minigpt/bpe.py",
    "kernels/cuda/README.md",
    "kernels/metal/kernels.py",
    "results/README.md",
    "tests/test_docs.py",
}


@pytest.fixture
def rw():
    return Rewriter(tracked=TRACKED)


# --------------------------------------------------------------- link targets


def test_readme_becomes_the_index(rw):
    assert rw.rewrite_link("lessons/13_triton.md", "../README.md") == "index.md"
    assert rw.rewrite_link("README.md", "README.md") == "index.md"


def test_lesson_links_stay_relative(rw):
    """Lessons are flat in docs/, so cross-links keep their bare filename."""
    assert rw.rewrite_link("lessons/00_setup.md", "01_tokenization.md") == "01_tokenization.md"
    assert rw.rewrite_link("README.md", "lessons/01_tokenization.md") == "01_tokenization.md"


def test_results_dir_and_readme_both_land_on_the_results_page(rw):
    assert rw.rewrite_link("README.md", "results/README.md") == "results.md"
    assert rw.rewrite_link("lessons/00_setup.md", "../results/") == "results.md"


def test_out_of_site_links_go_to_github(rw):
    got = rw.rewrite_link("lessons/12_gpu_kernels.md", "../kernels/cuda/README.md")
    assert got == ("https://github.com/bobo-coding/ai-learning/blob/main/"
                   "kernels/cuda/README.md")


def test_anchors_are_preserved(rw):
    assert rw.rewrite_link("README.md", "lessons/01_tokenization.md#why-byte-level") \
        == "01_tokenization.md#why-byte-level"
    assert rw.rewrite_link("README.md", "#measured-results") == "#measured-results"


def test_external_links_untouched(rw):
    for url in ["https://example.com/x", "http://example.com", "mailto:a@b.c"]:
        assert rw.rewrite_link("README.md", url) == url


def test_a_link_to_a_missing_file_is_an_error(rw):
    """Better a failed build than a 404 on the published site."""
    with pytest.raises(ValueError, match="does not exist"):
        rw.rewrite_link("README.md", "lessons/99_nope.md")


# ----------------------------------------------------------------- code spans


def test_code_spans_naming_real_files_become_links(rw):
    out = rw.rewrite("lessons/01_tokenization.md", "Read: `minigpt/bpe.py` now")
    assert out == ("Read: [`minigpt/bpe.py`]"
                   "(https://github.com/bobo-coding/ai-learning/blob/main/minigpt/bpe.py) now")


def test_code_spans_that_are_not_paths_are_left_alone(rw):
    for text in ["`torch.mps.compile_shader`", "`softmax(QK^T)`", "`n_embd`", "`a/b`"]:
        assert rw.rewrite("README.md", text) == text


def test_untracked_paths_are_left_alone(rw):
    """`out/` is gitignored, so linking to it would 404 on GitHub."""
    assert rw.rewrite("README.md", "`out/pretrain/history.json`") \
        == "`out/pretrain/history.json`"


def test_directory_code_spans_use_the_tree_url(rw):
    out = rw.rewrite("README.md", "see `kernels/metal/` for more")
    assert "tree/main/kernels/metal" in out


def test_fenced_code_blocks_are_never_rewritten(rw):
    text = (
        "before `minigpt/bpe.py`\n"
        "```\n"
        "minigpt/bpe.py    the tokenizer\n"
        "`minigpt/bpe.py`\n"
        "[x](lessons/00_setup.md)\n"
        "```\n"
        "after `minigpt/bpe.py`\n"
    )
    out = rw.rewrite("README.md", text)
    body = out.split("```")[1]
    assert "github.com" not in body, "fenced content must be left verbatim"
    assert out.count("github.com") == 2, "both prose mentions should be linked"


def test_code_spans_inside_a_link_label_are_not_double_wrapped(rw):
    text = "[`minigpt/bpe.py`](lessons/00_setup.md)"
    out = rw.rewrite("README.md", text)
    assert out == "[`minigpt/bpe.py`](00_setup.md)"
    assert out.count("](") == 1


def test_indented_fences_are_recognised(rw):
    text = "  ```\n  `minigpt/bpe.py`\n  ```\n"
    assert "github.com" not in rw.rewrite("README.md", text)


# ------------------------------------------------------- the real build output


def test_build_produces_every_expected_page(tmp_path):
    out = tmp_path / "docs"
    written = build(out, verbose=False)
    assert {p.name for p in written} == {dst for _, dst in PAGES}
    assert (out / "index.md").exists()
    assert (out / "00_setup.md").exists()
    assert (out / "13_triton.md").exists()
    assert (out / "results.md").exists()
    assert (out / "stylesheets" / "extra.css").exists()


def test_every_lesson_is_published():
    """A new lesson file must not silently miss the site."""
    lessons = {p.name for p in (ROOT / "lessons").glob("*.md")}
    published = {dst for src, dst in PAGES if src.startswith("lessons/")}
    assert lessons == published


def test_every_lesson_is_in_the_nav():
    """mkdocs --strict warns on omitted files; this makes it a hard failure."""
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for _, dst in PAGES:
        assert re.search(rf":\s*{re.escape(dst)}\s*$", nav, re.M), f"{dst} missing from nav"


def test_build_output_has_no_unrewritten_markdown_links(tmp_path):
    out = tmp_path / "docs"
    build(out, verbose=False)
    for page in out.glob("*.md"):
        text = page.read_text(encoding="utf-8")
        in_fence = False
        for line in text.splitlines():
            if line.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for m in re.finditer(r"\[[^\]]*\]\(([^)\s]+)\)", line):
                t = m.group(1)
                if t.startswith(("http", "#")):
                    continue
                # anything left must be a sibling page in docs/
                assert (out / t.split("#")[0]).exists(), f"{page.name}: dangling {t!r}"


# ---------------------------------------------------------- the link checker


def test_link_checker_accepts_a_good_site(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "index.html").write_text('<a href="../b/">b</a>')
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "index.html").write_text('<a href="../a/">a</a>')
    checked, broken = check(tmp_path)
    assert checked == 2 and broken == []


def test_link_checker_catches_a_dangling_link(tmp_path):
    (tmp_path / "index.html").write_text('<a href="missing/">x</a>')
    _, broken = check(tmp_path)
    assert len(broken) == 1 and "missing/" in broken[0]


def test_link_checker_ignores_404_page(tmp_path):
    (tmp_path / "index.html").write_text("<p>ok</p>")
    (tmp_path / "404.html").write_text('<link href="/site-root/absolute.css">')
    _, broken = check(tmp_path)
    assert broken == []


def test_link_checker_ignores_external_and_anchor_hrefs(tmp_path):
    (tmp_path / "index.html").write_text(
        '<a href="https://x.com">x</a><a href="#top">t</a><a href="mailto:a@b">m</a>'
    )
    checked, broken = check(tmp_path)
    assert checked == 0 and broken == []


def test_link_checker_rejects_a_missing_directory(tmp_path):
    with pytest.raises(SystemExit):
        check(tmp_path / "nope")
