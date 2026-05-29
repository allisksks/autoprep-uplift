"""
uplift/__init__.py
"""

from .pipeline import UpliftPipeline, fit_preprocess, apply_preprocess
from .metrics import uplift_at_k, auuc, qini_coefficient, evaluate, evaluate_all
from .validation import (
    check_randomization,
    check_leakage,
    permutation_test,
    learning_curves,
    repeated_cv,
    full_validation_report,
)
from .ensemble import UpliftEnsemble

__all__ = [
    'UpliftPipeline',
    'fit_preprocess',
    'apply_preprocess',
    'uplift_at_k',
    'auuc',
    'qini_coefficient',
    'evaluate',
    'evaluate_all',
    'check_randomization',
    'check_leakage',
    'permutation_test',
    'learning_curves',
    'repeated_cv',
    'full_validation_report',
    'UpliftEnsemble',
]