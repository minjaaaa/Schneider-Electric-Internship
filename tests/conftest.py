"""Zajednička podešavanja za testove: src/ na putanji, putanje do evaluacionih fajlova."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def facts() -> dict:
    return json.loads((ROOT / "eval" / "expected_facts.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def questions() -> list[dict]:
    return json.loads((ROOT / "eval" / "eval_questions.json").read_text(encoding="utf-8"))
