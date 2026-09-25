# tests/test_version.py
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import catlab

ROOT = os.path.join(os.path.dirname(__file__), '..')


def test_package_version_matches_citation_cff():
    """Regression: catlab.__version__ said 1.1.0 while the release was 3.5.5."""
    with open(os.path.join(ROOT, "CITATION.cff"), encoding="utf-8") as f:
        m = re.search(r"^version:\s*['\"]?([^'\"\s]+)", f.read(), re.MULTILINE)
    assert m, "no version field in CITATION.cff"
    assert catlab.__version__ == m.group(1)


def test_readme_badge_matches_package_version():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        m = re.search(r"badge/version-v([0-9][^-]*)-", f.read())
    assert m, "no version badge in README.md"
    assert catlab.__version__ == m.group(1)


def test_changelog_has_section_for_package_version():
    """The release workflow extracts '## v<version> ...' from CHANGELOG.md;
    an absent heading gives an empty release body."""
    with open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8") as f:
        text = f.read()
    assert re.search(rf"^## v{re.escape(catlab.__version__)}( |$)", text, re.MULTILINE)
