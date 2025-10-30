"""
Example script for analyzing profiling data from EdgeVolution NAS runs.

This script demonstrates how to load and analyze the profiling statistics
generated during a NAS run.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def load_profiling_data(results_dir):
    """
    Load profiling statistics from a results directory.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        Tuple of (detailed_stats, summary_stats)
    """
    results_path = Path(results_dir)
    
    stats_file = results_path / "profiling_stats.json"
    summary_file = results_path / "profiling_summary.json"
    
    with open(stats_file, 'r') as f:
        detailed_stats = json.load(f)
    
    with open(summary_file, 'r') as f:
        summary_stats = json.load(f)
    
    return detailed_stats, summary_stats


def analyze_generation_times(detailed_stats):
    """
    Analyze generation-level timing information.
    
    Args:
        detailed_stats: Detailed profiling statistics dictionary
        
    Returns:
        DataFrame with generation timing analysis
    """
    generations = detailed_stats['generations']
    
    data = []
    for gen_num, gen_data in generations.items():
        row = {
            'generation': int(gen_num),
            'total_duration': gen_data.get('total_duration', 0)
        }
        
        # Add phase times
        for phase_name, phase_time in gen_data.get('phases', {}).items():
            row[phase_name] = phase_time
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df.sort_values('generation')


def analyze_model_operations(detailed_stats, operation_type):
    """
    Analyze specific model operations across all generations.
    
    Args:
        detailed_stats: Detailed profiling statistics dictionary
        operation_type: Type of operation (e.g., 'training', 'deployment', 'translation')
        
    Returns:
        DataFrame with model operation timing analysis
    """
    generations = detailed_stats['generations']
    
    data = []
    for gen_num, gen_data in generations.items():
        for model_name, operations in gen_data.get('models', {}).items():
            if operation_type in operations:
                data.append({
                    'generation': int(gen_num),
                    'model': model_name,
                    operation_type: operations[operation_type]
                })
    
    df = pd.DataFrame(data)
    return df


def print_summary_report(summary_stats):
    """
    Print a formatted summary report.
    
    Args:
        summary_stats: Summary statistics dictionary
    """
    print("=" * 80)
    print("EdgeVolution NAS - Profiling Summary Report")
    print("=" * 80)
    print()
    
    # Overall statistics
    print("OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total Duration:        {summary_stats['overall_duration']:.2f} seconds")
    print(f"                       ({summary_stats['overall_duration']/3600:.2f} hours)")
    print(f"Number of Generations: {summary_stats['num_generations']}")
    print()
    
    # Generation statistics
    gen_stats = summary_stats['generation_durations']
    print("GENERATION STATISTICS")
    print("-" * 80)
    print(f"Average per Generation: {gen_stats['mean']:.2f} seconds")
    print(f"Fastest Generation:     {gen_stats['min']:.2f} seconds")
    print(f"Slowest Generation:     {gen_stats['max']:.2f} seconds")
    print()
    
    # Phase statistics
    print("PHASE BREAKDOWN (Average across all generations)")
    print("-" * 80)
    phase_stats = summary_stats['phase_durations']
    
    # Sort phases by mean duration
    sorted_phases = sorted(
        phase_stats.items(), 
        key=lambda x: x[1]['mean'] if x[1]['mean'] is not None else 0, 
        reverse=True
    )
    
    for phase_name, stats in sorted_phases:
        if stats['mean'] is not None:
            percentage = (stats['mean'] / gen_stats['mean']) * 100 if gen_stats['mean'] else 0
            print(f"{phase_name:35} {stats['mean']:10.2f}s  ({percentage:5.1f}% of generation)")
    
    print("=" * 80)


def export_to_csv(detailed_stats, output_dir):
    """
    Export profiling data to CSV files for further analysis.
    
    Args:
        detailed_stats: Detailed profiling statistics dictionary
        output_dir: Directory to save CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Export generation times
    gen_df = analyze_generation_times(detailed_stats)
    gen_df.to_csv(output_path / 'generation_times.csv', index=False)
    print(f"Exported generation times to {output_path / 'generation_times.csv'}")
    
    # Export model operations
    operations = ['translation', 'tflite_conversion', 'training', 'deployment']
    for op in operations:
        try:
            op_df = analyze_model_operations(detailed_stats, op)
            if not op_df.empty:
                op_df.to_csv(output_path / f'model_{op}_times.csv', index=False)
                print(f"Exported {op} times to {output_path / f'model_{op}_times.csv'}")
        except Exception as e:
            print(f"Could not export {op} times: {e}")


def compare_phases_across_generations(detailed_stats):
    """
    Create a comparison of how phase times change across generations.
    
    Args:
        detailed_stats: Detailed profiling statistics dictionary
        
    Returns:
        DataFrame with phase times across generations
    """
    generations = detailed_stats['generations']
    
    # Collect all unique phase names
    all_phases = set()
    for gen_data in generations.values():
        all_phases.update(gen_data.get('phases', {}).keys())
    
    # Build comparison table
    data = []
    for gen_num in sorted([int(g) for g in generations.keys()]):
        gen_data = generations[str(gen_num)]
        row = {'generation': gen_num}
        
        for phase in all_phases:
            row[phase] = gen_data.get('phases', {}).get(phase, None)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_profiling.py <results_directory> [output_csv_dir]")
        print("\nExample:")
        print("  python analyze_profiling.py Results/edgevolution_20241015-120000_cifar10")
        print("  python analyze_profiling.py Results/edgevolution_20241015-120000_cifar10 ./analysis_output")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    try:
        # Load data
        print(f"Loading profiling data from {results_dir}...")
        detailed_stats, summary_stats = load_profiling_data(results_dir)
        
        # Print summary report
        print_summary_report(summary_stats)
        
        # Print additional analysis
        print("\nDETAILED PHASE COMPARISON ACROSS GENERATIONS")
        print("-" * 80)
        phase_comparison = compare_phases_across_generations(detailed_stats)
        print(phase_comparison.to_string(index=False))
        print()
        
        # Export to CSV if requested
        if len(sys.argv) >= 3:
            output_dir = sys.argv[2]
            print(f"\nExporting data to CSV files in {output_dir}...")
            export_to_csv(detailed_stats, output_dir)
            
            # Also export phase comparison
            phase_comparison.to_csv(Path(output_dir) / 'phase_comparison.csv', index=False)
            print(f"Exported phase comparison to {Path(output_dir) / 'phase_comparison.csv'}")
        
        print("\nAnalysis complete!")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find profiling files in {results_dir}")
        print(f"Make sure the directory contains profiling_stats.json and profiling_summary.json")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
