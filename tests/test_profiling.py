"""
Test script to verify profiling functionality.

This script tests the profiling infrastructure without running a full NAS experiment.
"""

import sys
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_architecture_search.utils.profiling import (
    ProfilingStats, 
    time_phase, 
    time_model_operation,
    calculate_summary_statistics
)


def simulate_nas_run():
    """Simulate a simple NAS run to test profiling."""
    
    print("Testing EdgeVolution Profiling Infrastructure")
    print("=" * 60)
    
    # Initialize profiling
    profiling_stats = ProfilingStats()
    profiling_stats.start_overall_run()
    
    num_generations = 3
    models_per_generation = 4
    
    # Simulate generations
    for gen in range(1, num_generations + 1):
        print(f"\nGeneration {gen}/{num_generations}")
        profiling_stats.start_generation(gen)
        
        # Simulate prepare phase
        print("  - Preparing generation...")
        with time_phase(profiling_stats, gen, "prepare_generation"):
            time.sleep(0.5)
        
        # Simulate translation and conversion
        print("  - Translating and converting models...")
        with time_phase(profiling_stats, gen, "translation_and_conversion_parallel"):
            for model_idx in range(models_per_generation):
                model_name = f"model_{gen}_{model_idx}"
                
                # Simulate translation
                with time_model_operation(profiling_stats, gen, model_name, "translation"):
                    time.sleep(0.1)
                
                # Simulate TFLite conversion
                with time_model_operation(profiling_stats, gen, model_name, "tflite_conversion"):
                    time.sleep(0.05)
        
        # Simulate memory evaluation
        print("  - Evaluating memory footprint...")
        with time_phase(profiling_stats, gen, "evaluate_memory_footprint"):
            time.sleep(0.1)
        
        # Simulate training
        print("  - Training models...")
        with time_phase(profiling_stats, gen, "train_neural_networks"):
            for model_idx in range(models_per_generation):
                model_name = f"model_{gen}_{model_idx}"
                
                # Simulate training
                with time_model_operation(profiling_stats, gen, model_name, "training"):
                    time.sleep(0.2)
        
        # Simulate deployment (only for first 2 models)
        print("  - Deploying to hardware...")
        for model_idx in range(2):
            model_name = f"model_{gen}_{model_idx}"
            with time_model_operation(profiling_stats, gen, model_name, "deployment"):
                time.sleep(0.15)
        
        # Simulate selection
        print("  - Selecting best models...")
        with time_phase(profiling_stats, gen, "selection"):
            time.sleep(0.1)
        
        # Simulate crossover and mutation (except last generation)
        if gen < num_generations:
            print("  - Crossover...")
            with time_phase(profiling_stats, gen, "crossover"):
                time.sleep(0.08)
            
            print("  - Mutation...")
            with time_phase(profiling_stats, gen, "mutation"):
                time.sleep(0.05)
        
        profiling_stats.end_generation(gen)
    
    # Finalize profiling
    profiling_stats.end_overall_run()
    
    return profiling_stats


def verify_profiling_data(profiling_stats):
    """Verify that profiling data was collected correctly."""
    
    print("\n" + "=" * 60)
    print("Verification Results")
    print("=" * 60)
    
    stats = profiling_stats.get_stats()
    
    # Check overall timing
    assert stats["overall_start_time"] is not None, "Overall start time not recorded"
    assert stats["overall_end_time"] is not None, "Overall end time not recorded"
    assert stats["overall_duration"] is not None, "Overall duration not calculated"
    assert stats["overall_duration"] > 0, "Overall duration should be positive"
    print("✓ Overall timing recorded correctly")
    
    # Check generation data
    num_generations = len(stats["generations"])
    assert num_generations > 0, "No generation data recorded"
    print(f"✓ Recorded data for {num_generations} generations")
    
    # Check phase data
    for gen_num, gen_data in stats["generations"].items():
        assert "phases" in gen_data, f"No phase data for generation {gen_num}"
        assert len(gen_data["phases"]) > 0, f"No phases recorded for generation {gen_num}"
        assert "total_duration" in gen_data, f"No total duration for generation {gen_num}"
    print("✓ Phase timing recorded for all generations")
    
    # Check model data
    total_models = 0
    for gen_data in stats["generations"].values():
        if "models" in gen_data:
            total_models += len(gen_data["models"])
    assert total_models > 0, "No model operation data recorded"
    print(f"✓ Recorded operations for {total_models} models")
    
    # Check specific operations
    for gen_data in stats["generations"].values():
        for model_name, operations in gen_data.get("models", {}).items():
            assert "translation" in operations, f"Translation not recorded for {model_name}"
            assert "tflite_conversion" in operations, f"TFLite conversion not recorded for {model_name}"
            # Note: Not all models have training/deployment in this test
    print("✓ Model operations (translation, conversion) recorded correctly")
    
    print("\n✓ All verification checks passed!")


def test_save_and_load(profiling_stats, temp_dir="./test_profiling_output"):
    """Test saving and loading profiling data."""
    
    print("\n" + "=" * 60)
    print("Testing Save/Load Functionality")
    print("=" * 60)
    
    # Create temporary directory
    temp_path = Path(temp_dir)
    temp_path.mkdir(exist_ok=True)
    
    # Save profiling data
    stats_file = temp_path / "test_profiling_stats.json"
    profiling_stats.save_to_json(str(stats_file))
    print(f"✓ Saved profiling data to {stats_file}")
    
    # Verify file exists and is valid JSON
    assert stats_file.exists(), "Profiling stats file not created"
    with open(stats_file, 'r') as f:
        data = json.load(f)
    assert "overall_duration" in data, "Invalid JSON structure"
    print("✓ File created with valid JSON structure")
    
    # Load profiling data
    loaded_stats = ProfilingStats.load_from_json(str(stats_file))
    assert loaded_stats.get_stats() == profiling_stats.get_stats(), "Loaded data doesn't match"
    print("✓ Data loaded correctly")
    
    # Test summary statistics
    summary = calculate_summary_statistics(profiling_stats)
    summary_file = temp_path / "test_profiling_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary statistics saved to {summary_file}")
    
    # Verify summary structure
    assert "overall_duration" in summary, "Summary missing overall_duration"
    assert "num_generations" in summary, "Summary missing num_generations"
    assert "generation_durations" in summary, "Summary missing generation_durations"
    assert "phase_durations" in summary, "Summary missing phase_durations"
    print("✓ Summary statistics structure correct")
    
    print(f"\nTest files saved in: {temp_path}")
    print("You can inspect these files to see the profiling output format")


def print_sample_output(profiling_stats):
    """Print a sample of the profiling output."""
    
    print("\n" + "=" * 60)
    print("Sample Profiling Output")
    print("=" * 60)
    
    stats = profiling_stats.get_stats()
    
    print(f"\nOverall Duration: {stats['overall_duration']:.2f} seconds")
    print(f"Number of Generations: {len(stats['generations'])}")
    
    # Show first generation details
    gen_1 = stats['generations']['1']
    print(f"\nGeneration 1 Duration: {gen_1['total_duration']:.2f} seconds")
    print("\nPhases:")
    for phase_name, duration in gen_1['phases'].items():
        print(f"  {phase_name:40} {duration:8.3f}s")
    
    # Show first model details
    if gen_1.get('models'):
        first_model = list(gen_1['models'].keys())[0]
        print(f"\nModel: {first_model}")
        for operation, duration in gen_1['models'][first_model].items():
            print(f"  {operation:40} {duration:8.3f}s")
    
    # Show summary statistics
    summary = calculate_summary_statistics(profiling_stats)
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    gen_stats = summary['generation_durations']
    print(f"\nGeneration Durations:")
    print(f"  Mean: {gen_stats['mean']:.2f}s")
    print(f"  Min:  {gen_stats['min']:.2f}s")
    print(f"  Max:  {gen_stats['max']:.2f}s")
    
    print(f"\nPhase Durations (Average):")
    for phase_name, phase_stats in summary['phase_durations'].items():
        if phase_stats['mean'] is not None:
            print(f"  {phase_name:40} {phase_stats['mean']:8.3f}s")


if __name__ == "__main__":
    print("EdgeVolution Profiling Test Suite")
    print("=" * 60)
    print("\nThis test simulates a small NAS run to verify profiling works correctly.")
    print("No actual models will be created or trained.\n")
    
    try:
        # Run simulation
        profiling_stats = simulate_nas_run()
        
        # Verify data
        verify_profiling_data(profiling_stats)
        
        # Test save/load
        test_save_and_load(profiling_stats)
        
        # Print sample output
        print_sample_output(profiling_stats)
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        print("\nThe profiling infrastructure is working correctly.")
        print("You can now use it in your actual NAS runs.")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
