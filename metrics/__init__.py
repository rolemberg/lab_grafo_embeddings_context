"""Metricas de avaliacao para os experimentos."""

from .hallucination import check_answer, classify_answer
from .significance import wilson_interval, accuracy_with_ci, mcnemar_test, mcnemar_by_segment

__all__ = [
    "check_answer", "classify_answer",
    "wilson_interval", "accuracy_with_ci", "mcnemar_test", "mcnemar_by_segment",
]