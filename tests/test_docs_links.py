"""Every relative link and image in the README and the docs pages points at a file that
exists, so moving sections into docs/ breaks nothing."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "SETUP.md",
         ROOT / "CONTRIBUTIONS.md"]
LINK = re.compile(r"\]\(([^)\s]+)\)|<img[^>]*\bsrc=\"([^\"]+)\"")


def links(page: Path) -> list[str]:
    text = page.read_text(encoding="utf-8")
    # code blocks hold commands, not links
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = "\n".join(line for line in text.splitlines() if not line.startswith("    "))
    return [a or b for a, b in LINK.findall(text)]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(page):
    broken = []
    for target in links(page):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue  # a web address or an anchor in the same page
        path = (page.parent / target.split("#", 1)[0]).resolve()
        if not path.exists():
            broken.append(target)
    assert not broken, f"{page.name}: broken links {broken}"
