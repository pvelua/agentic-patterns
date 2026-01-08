"""Multi-Agent Collaboration pattern - Three different collaboration models."""

from .run import (
    run_research_analysis,
    run_product_launch,
    run_code_review,
    compare_models
)
from .config import MultiAgentCollabConfig

__all__ = [
    'run_research_analysis',
    'run_product_launch',
    'run_code_review',
    'compare_models',
    'MultiAgentCollabConfig',
]
