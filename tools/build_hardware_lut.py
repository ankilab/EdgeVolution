"""
Build a Hardware Lookup Table (LUT) from a completed NAS run.

Reads profiling results (chromosome.json + results.json) from a previous
run directory, trains per-metric cost models, and saves a reusable LUT.

Usage::

    python tools/build_hardware_lut.py \
        --results-dir Results/edgevolution_20260311-214851_speech_commands \
        --board-snr <YOUR_BOARD_SNR> \
        --output hardware_luts/nrf52840dk_speech_commands/ \
        --cross-validate
"""

import argparse
import json
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_architecture_search.src.search_space_registry import SearchSpaceRegistry
from neural_architecture_search.src.hardware_lut import HardwareLUT


def main():
    parser = argparse.ArgumentParser(
        description="Build a Hardware LUT from a completed NAS run.",
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Path to the NAS results directory (contains Generation_*, search_space.json).",
    )
    parser.add_argument(
        "--board-snr", required=True,
        help="Board serial number to extract metrics for.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for the Hardware LUT.",
    )
    parser.add_argument(
        "--mode", default="full", choices=["full", "layerwise"],
        help="Prediction mode: 'full' (whole-architecture) or 'layerwise' (per-layer). Default: full.",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100,
        help="Number of trees for Random Forest models. Default: 100.",
    )
    parser.add_argument(
        "--cross-validate", action="store_true",
        help="Run 5-fold cross-validation and print R²/MAE per metric.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    # Load search space registry from the results directory
    search_space_path = os.path.join(results_dir, "search_space.json")
    if not os.path.exists(search_space_path):
        print(f"ERROR: search_space.json not found in {results_dir}")
        sys.exit(1)

    with open(search_space_path) as f:
        search_space_config = json.load(f)

    registry = SearchSpaceRegistry.from_dict(search_space_config, validate=False)

    print(f"Building Hardware LUT (mode={args.mode})...")
    print(f"  Results: {results_dir}")
    print(f"  Board SNR: {args.board_snr}")
    print()

    lut = HardwareLUT.build_from_results(
        results_dir,
        args.board_snr,
        registry,
        mode=args.mode,
        n_estimators=args.n_estimators,
    )

    print(f"\nLUT built: {len(lut.models)} metrics, "
          f"{lut.metadata['total_samples']} total samples")

    # Cross-validation
    if args.cross_validate:
        print("\nCross-validation (5-fold):")
        cv_results = lut.cross_validate(
            results_dir, args.board_snr, registry, n_folds=5,
        )
        for metric, scores in cv_results.items():
            print(f"  {metric}: R²={scores['r2']:.4f}, MAE={scores['mae']:.4f}")
        if not cv_results:
            print("  (no metrics had enough samples for cross-validation)")

    # Save
    lut.save(args.output)
    print(f"\nHardware LUT saved to: {args.output}")
    print(f"\nTo use in a NAS run:")
    print(f"  python main.py ... "
          f"hardware_lut.enabled.value=true "
          f"hardware_lut.path.value={args.output}")


if __name__ == "__main__":
    main()
