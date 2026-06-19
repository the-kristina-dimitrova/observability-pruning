# src/methods/__init__.py
from .observability import ObservabilityScorer, prune_by_observability
from .hybrid_pruning import prune_hybrid, hybrid_score_analysis
try:
    from .magnitude_pruning import prune_by_magnitude, compute_magnitude_scores
    from .lottery_ticket import LotteryTicketPruner
except ModuleNotFoundError:
    pass