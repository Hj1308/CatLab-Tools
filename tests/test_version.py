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
