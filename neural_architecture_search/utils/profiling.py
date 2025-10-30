"""
Profiling utilities for EdgeVolution NAS.

This module provides timing and profiling capabilities for tracking
the performance of different stages in the neural architecture search process.
"""

import time
import json
from typing import Dict, Any, Optional
from contextlib import contextmanager


class ProfilingStats:
    """
    Class to collect and manage profiling statistics during NAS execution.
    
    Tracks timing information for:
    - Overall run time
    - Per-generation times
    - Individual phase times (prepare, train, deploy, selection, etc.)
    - Per-model operations (translation, conversion, training, deployment)
    """
    
    def __init__(self, save_dir: Optional[str] = None, verbose: bool = True):
        """
        Initialize profiling statistics.
        
        Args:
            save_dir: Directory to save incremental profiling data (optional)
            verbose: If True, print profiling information to console
        """
        self.stats: Dict[str, Any] = {
            "overall_start_time": None,
            "overall_end_time": None,
            "overall_duration": None,
            "generations": {}
        }
        self.save_dir = save_dir
        self.verbose = verbose
        
    def start_overall_run(self):
        """Mark the start of the overall NAS run."""
        self.stats["overall_start_time"] = time.time()
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Starting NAS run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")
        
    def end_overall_run(self):
        """Mark the end of the overall NAS run and calculate duration."""
        self.stats["overall_end_time"] = time.time()
        if self.stats["overall_start_time"] is not None:
            self.stats["overall_duration"] = self.stats["overall_end_time"] - self.stats["overall_start_time"]
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"NAS run completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Total duration: {self.stats['overall_duration']:.2f} seconds ({self.stats['overall_duration']/3600:.2f} hours)")
                print(f"{'='*80}\n")
    
    def start_generation(self, generation: int):
        """
        Initialize timing data for a generation.
        
        Args:
            generation: Generation number
        """
        if generation not in self.stats["generations"]:
            self.stats["generations"][generation] = {
                "start_time": time.time(),
                "end_time": None,
                "total_duration": None,
                "phases": {},
                "models": {}
            }
        else:
            self.stats["generations"][generation]["start_time"] = time.time()
    
    def end_generation(self, generation: int):
        """
        Finalize timing data for a generation and save incrementally.
        
        Args:
            generation: Generation number
        """
        if generation in self.stats["generations"]:
            self.stats["generations"][generation]["end_time"] = time.time()
            start = self.stats["generations"][generation]["start_time"]
            end = self.stats["generations"][generation]["end_time"]
            self.stats["generations"][generation]["total_duration"] = end - start
            
            # Print generation summary if verbose
            if self.verbose:
                duration = self.stats["generations"][generation]["total_duration"]
                print(f"\n{'-'*80}")
                print(f"Generation {generation} completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
                
                # Print phase breakdown
                phases = self.stats["generations"][generation].get("phases", {})
                if phases:
                    print(f"\nPhase breakdown:")
                    for phase_name, phase_time in sorted(phases.items(), key=lambda x: x[1], reverse=True):
                        percentage = (phase_time / duration * 100) if duration > 0 else 0
                        print(f"  {phase_name:40} {phase_time:8.2f}s ({percentage:5.1f}%)")
                
                # Print model statistics
                models = self.stats["generations"][generation].get("models", {})
                if models:
                    # Aggregate model operation times
                    operation_times = {}
                    for model_data in models.values():
                        for op_name, op_time in model_data.items():
                            if op_name not in operation_times:
                                operation_times[op_name] = []
                            operation_times[op_name].append(op_time)
                    
                    if operation_times:
                        print(f"\nModel operations (average across {len(models)} models):")
                        for op_name, times in sorted(operation_times.items()):
                            avg_time = sum(times) / len(times)
                            print(f"  {op_name:40} {avg_time:8.2f}s avg")
                
                print(f"{'-'*80}\n")
            
            # Save incrementally after each generation
            if self.save_dir:
                self._save_incremental(generation)
    
    def record_phase_time(self, generation: int, phase_name: str, duration: float):
        """
        Record the duration of a specific phase within a generation.
        
        Args:
            generation: Generation number
            phase_name: Name of the phase (e.g., 'prepare', 'train', 'deploy')
            duration: Duration in seconds
        """
        if generation not in self.stats["generations"]:
            self.start_generation(generation)
        
        self.stats["generations"][generation]["phases"][phase_name] = duration
        
        if self.verbose:
            print(f"  Phase '{phase_name}' completed in {duration:.2f} seconds")
    
    def record_model_operation(self, generation: int, model_name: str, 
                               operation: str, duration: float):
        """
        Record timing for individual model operations.
        
        Args:
            generation: Generation number
            model_name: Name of the individual/model
            operation: Type of operation (e.g., 'translation', 'conversion', 'training', 'deployment')
            duration: Duration in seconds
        """
        if generation not in self.stats["generations"]:
            self.start_generation(generation)
        
        if model_name not in self.stats["generations"][generation]["models"]:
            self.stats["generations"][generation]["models"][model_name] = {}
        
        self.stats["generations"][generation]["models"][model_name][operation] = duration
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get the complete profiling statistics.
        
        Returns:
            Dictionary containing all profiling data
        """
        return self.stats
    
    def get_generation_summary(self, generation: int) -> Optional[Dict[str, Any]]:
        """
        Get summary statistics for a specific generation.
        
        Args:
            generation: Generation number
            
        Returns:
            Dictionary with generation statistics or None if not found
        """
        return self.stats["generations"].get(generation)
    
    def save_to_json(self, filepath: str):
        """
        Save profiling statistics to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
        if self.verbose:
            print(f"Profiling statistics saved to {filepath}")
    
    def _save_incremental(self, generation: int):
        """
        Save profiling data incrementally after each generation.
        
        Args:
            generation: Generation number just completed
        """
        if not self.save_dir:
            return
        
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save complete stats so far
        filepath = os.path.join(self.save_dir, "profiling_stats_incremental.json")
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'ProfilingStats':
        """
        Load profiling statistics from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            ProfilingStats object with loaded data
        """
        instance = cls()
        with open(filepath, 'r') as f:
            instance.stats = json.load(f)
        return instance


@contextmanager
def time_phase(profiling_stats: ProfilingStats, generation: int, phase_name: str):
    """
    Context manager for timing a phase within a generation.
    
    Usage:
        with time_phase(profiling_stats, generation=1, phase_name="training"):
            # code to time
            pass
    
    Args:
        profiling_stats: ProfilingStats instance
        generation: Generation number
        phase_name: Name of the phase
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        profiling_stats.record_phase_time(generation, phase_name, duration)


@contextmanager
def time_model_operation(profiling_stats: ProfilingStats, generation: int, 
                         model_name: str, operation: str):
    """
    Context manager for timing individual model operations.
    
    Usage:
        with time_model_operation(profiling_stats, generation=1, model_name="model_1", operation="training"):
            # code to time
            pass
    
    Args:
        profiling_stats: ProfilingStats instance
        generation: Generation number
        model_name: Name of the model
        operation: Type of operation
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        profiling_stats.record_model_operation(generation, model_name, operation, duration)


def calculate_summary_statistics(profiling_stats: ProfilingStats) -> Dict[str, Any]:
    """
    Calculate summary statistics across all generations.
    
    Args:
        profiling_stats: ProfilingStats instance
        
    Returns:
        Dictionary with summary statistics including means, mins, maxs
    """
    summary = {
        "overall_duration": profiling_stats.stats.get("overall_duration"),
        "num_generations": len(profiling_stats.stats["generations"]),
        "generation_durations": {
            "mean": None,
            "min": None,
            "max": None,
            "total": None
        },
        "phase_durations": {}
    }
    
    # Calculate generation duration statistics
    gen_durations = []
    for gen_data in profiling_stats.stats["generations"].values():
        if gen_data.get("total_duration") is not None:
            gen_durations.append(gen_data["total_duration"])
    
    if gen_durations:
        summary["generation_durations"]["mean"] = sum(gen_durations) / len(gen_durations)
        summary["generation_durations"]["min"] = min(gen_durations)
        summary["generation_durations"]["max"] = max(gen_durations)
        summary["generation_durations"]["total"] = sum(gen_durations)
    
    # Calculate phase duration statistics across all generations
    phase_times = {}
    for gen_data in profiling_stats.stats["generations"].values():
        for phase_name, duration in gen_data.get("phases", {}).items():
            if phase_name not in phase_times:
                phase_times[phase_name] = []
            phase_times[phase_name].append(duration)
    
    for phase_name, durations in phase_times.items():
        summary["phase_durations"][phase_name] = {
            "mean": sum(durations) / len(durations) if durations else None,
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "total": sum(durations) if durations else None,
            "count": len(durations)
        }
    
    return summary
