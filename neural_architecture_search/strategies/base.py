"""
Base classes for pluggable search strategies.

The ask/tell interface separates "what to try" from "how to evaluate":

    for each generation:
        candidates = strategy.ask(n)
        results = pipeline.evaluate(candidates)
        strategy.tell(results)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    """A candidate architecture proposed by a search strategy."""
    name: str                                    # e.g. "happy_panda_3"
    chromosome: List[Dict[str, Any]]             # genotype (list of gene dicts)
    metadata: Dict[str, Any] = field(default_factory=dict)  # strategy-specific


@dataclass
class EvaluationResult:
    """Result of evaluating a candidate through the pipeline."""
    name: str
    chromosome: List[Dict[str, Any]]
    val_acc: float
    memory_footprint_tflite: int
    fitness: float
    energy_consumption: Optional[float] = None
    inference_time: Optional[float] = None
    rom_usage: Optional[int] = None
    surrogate_predicted: bool = False
    hw_surrogate_predicted: bool = False
    hw_lut_predicted: bool = False
    error: bool = False


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    @abstractmethod
    def ask(self, n_candidates: int) -> List[Candidate]:
        """Propose n candidate architectures for evaluation."""
        ...

    @abstractmethod
    def tell(self, results: List[EvaluationResult]) -> None:
        """Receive evaluation results and update internal state."""
        ...

    @abstractmethod
    def get_best(self, n: int = 1) -> List[EvaluationResult]:
        """Return the top-n results seen so far."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    def on_generation_start(self, generation: int) -> None:
        """Hook called at the start of each generation."""
        pass

    def on_generation_end(self, generation: int) -> None:
        """Hook called at the end of each generation."""
        pass
