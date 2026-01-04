"""Parallelization pattern - Concurrent execution of multiple LLM chains using RunnableParallel."""

from .run import run, compare_models
from .config import ParallelizationConfig

__all__ = [
    'run',
    'compare_models',
    'ParallelizationConfig',
]
