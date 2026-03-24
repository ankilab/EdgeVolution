"""CLI wrapper to regenerate the surrogate evaluation plot from existing CSV logs."""

import sys
from pathlib import Path

from neural_architecture_search.src.surrogate_model import SurrogateModel


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_surrogate.py <results_dir> [surrogate_type]")
        print("  surrogate_type: 'accuracy' (default), 'hardware', or 'both'")
        print("Example: python scripts/plot_surrogate.py Results/edgevolution_..._speech_commands")
        print("Example: python scripts/plot_surrogate.py Results/edgevolution_..._speech_commands hardware")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    surrogate_type = sys.argv[2] if len(sys.argv) > 2 else "both"

    dirs_to_plot = []
    if surrogate_type in ("accuracy", "both"):
        # Check new name first, then legacy name
        acc_dir = results_dir / "surrogate_accuracy"
        if not acc_dir.exists():
            acc_dir = results_dir / "surrogate"
        if acc_dir.exists():
            dirs_to_plot.append(acc_dir)
    if surrogate_type in ("hardware", "both"):
        hw_dir = results_dir / "surrogate_hardware"
        if hw_dir.exists():
            dirs_to_plot.append(hw_dir)

    if not dirs_to_plot:
        print(f"No surrogate directories found in {results_dir}")
        sys.exit(1)

    for surrogate_dir in dirs_to_plot:
        SurrogateModel.plot_evaluation(str(surrogate_dir))
        print(f"Saved plot to {surrogate_dir / 'surrogate_evaluation.png'}")
