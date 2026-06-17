"""Smoke-тест: пакет импортируется, окружение рабочее."""
import importlib

import numpy as np


def test_core_modules_import():
    for mod in [
        "uplift",
        "uplift.metrics",
        "uplift.pipeline",
        "uplift.ensemble",
        "uplift.validation",
    ]:
        assert importlib.import_module(mod) is not None


def test_numpy_sanity():
    assert np.array([1, 2, 3]).sum() == 6
