import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def small_sample():
    from airlock.sample import build_sample_pdf

    return build_sample_pdf(pages=120)


@pytest.fixture(scope="session")
def full_sample():
    from airlock.sample import build_sample_pdf

    return build_sample_pdf()  # 1,040 pages
