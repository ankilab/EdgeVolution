"""CLI wrapper to regenerate the surrogate evaluation plot from existing CSV logs."""

import sys
from pathlib import Path

from neural_architecture_search.src.surrogate_model import SurrogateModel


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_surrogate.py <results_dir>")
        print("Example: python scripts/plot_surrogate.py Results/edgevolution_..._speech_commands")
        sys.exit(1)

    surrogate_dir = Path(sys.argv[1]) / "surrogate"
    if not surrogate_dir.exists():
        print(f"No surrogate directory found at {surrogate_dir}")
        sys.exit(1)

    SurrogateModel.plot_evaluation(str(surrogate_dir))
    print(f"Saved plot to {surrogate_dir / 'surrogate_evaluation.png'}")
