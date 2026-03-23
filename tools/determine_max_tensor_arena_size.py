#!/usr/bin/env python3
"""
Determine the maximum available tensor arena size for a given board.

Builds firmware with a minimal (1 KB) tensor arena, reads the RAM report,
and computes how much RAM remains for the tensor arena.

Usage:
    python tools/determine_max_tensor_arena_size.py --board nrf52840dk_nrf52840
    python tools/determine_max_tensor_arena_size.py --board nrf52840dk_nrf52840 --safety-margin 4
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from update_tensor_arena_size import update_tensor_arena_size

# Total SRAM available per board (from Nordic datasheets)
BOARD_SRAM_KB = {
    "nrf52833dk_nrf52833": 128,
    "nrf52840dk_nrf52840": 256,
    "nrf5340dk_nrf5340_cpuapp": 512,
}

MAIN_FUNCTIONS_CPP = os.path.join(
    os.path.dirname(__file__), "..", "tflite", "edgevolution_tflite", "src", "main_functions.cpp"
)
TFLITE_DIR = os.path.join(os.path.dirname(__file__), "..", "tflite")


def get_current_arena_size_kb(cpp_path):
    """Read the current kTensorArenaSize from the .cpp file."""
    import re
    with open(cpp_path) as f:
        for line in f:
            m = re.search(r"constexpr\s+int\s+kTensorArenaSize\s*=\s*(\d+)\s*\*\s*1024", line)
            if m:
                return int(m.group(1))
    return None


def determine_max_tensor_arena_size(board_type, safety_margin_kb=4):
    """
    Determine max tensor arena size by building with minimal arena and reading RAM usage.

    Args:
        board_type: Board model string (e.g. "nrf52840dk_nrf52840")
        safety_margin_kb: Safety margin in KB to subtract from the computed max

    Returns:
        int: Maximum tensor arena size in KB
    """
    if board_type not in BOARD_SRAM_KB:
        print(f"Error: Unknown board type '{board_type}'.")
        print(f"Supported boards: {', '.join(BOARD_SRAM_KB.keys())}")
        return None

    total_sram_kb = BOARD_SRAM_KB[board_type]
    cpp_path = os.path.abspath(MAIN_FUNCTIONS_CPP)
    tflite_dir = os.path.abspath(TFLITE_DIR)

    # Save original arena size to restore later
    original_size_kb = get_current_arena_size_kb(cpp_path)
    if original_size_kb is None:
        print("Error: Could not read current kTensorArenaSize from main_functions.cpp")
        return None

    probe_size_kb = 1
    print(f"Board: {board_type} (total SRAM: {total_sram_kb} KB)")
    print(f"Setting tensor arena to {probe_size_kb} KB for probe build...")

    update_tensor_arena_size(cpp_path, probe_size_kb)

    try:
        # Build firmware
        build_dir = f"build-{board_type}"
        print(f"Building firmware (west build -b {board_type})...")
        subprocess.run(
            ["west", "build", "-b", board_type, "edgevolution_tflite/",
             "--build-dir", build_dir, "--pristine"],
            cwd=tflite_dir, check=True, capture_output=True, text=True,
        )

        # Generate RAM report
        print("Generating RAM report...")
        subprocess.run(
            ["west", "build", "--build-dir", build_dir, "-t", "ram_report"],
            cwd=tflite_dir, check=True, capture_output=True, text=True,
        )

        # Read RAM usage
        ram_json_path = os.path.join(tflite_dir, build_dir, "ram.json")
        with open(ram_json_path) as f:
            ram_data = json.load(f)

        base_ram_bytes = ram_data["total_size"]
        base_ram_kb = base_ram_bytes / 1024

        # Compute: overhead = base_ram - probe_arena, max_arena = total_sram - overhead - safety
        overhead_kb = base_ram_kb - probe_size_kb
        max_arena_kb = int(total_sram_kb - overhead_kb - safety_margin_kb)

        print(f"\nResults:")
        print(f"  Base RAM usage (with {probe_size_kb} KB arena): {base_ram_kb:.1f} KB")
        print(f"  Firmware overhead (excl. arena):                 {overhead_kb:.1f} KB")
        print(f"  Safety margin:                                   {safety_margin_kb} KB")
        print(f"  Max available tensor arena size:                 {max_arena_kb} KB")

        return max_arena_kb

    finally:
        # Restore original arena size
        print(f"\nRestoring original tensor arena size ({original_size_kb} KB)...")
        update_tensor_arena_size(cpp_path, original_size_kb)


def main():
    parser = argparse.ArgumentParser(
        description="Determine the maximum available tensor arena size for a given board."
    )
    parser.add_argument(
        "--board", type=str, required=True,
        help=f"Board model. Supported: {', '.join(BOARD_SRAM_KB.keys())}"
    )
    parser.add_argument(
        "--safety-margin", type=int, default=4,
        help="Safety margin in KB (default: 4)"
    )
    args = parser.parse_args()

    result = determine_max_tensor_arena_size(args.board, args.safety_margin)
    if result is None:
        sys.exit(1)

    print(f"\nRecommended config value: max_available_tensor_arena_size: {result}")


if __name__ == "__main__":
    main()
