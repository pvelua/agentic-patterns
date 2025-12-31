"""Utility for writing experiment results to log files."""
from pathlib import Path
from datetime import datetime
from typing import Any
import json
from .config import settings


class OutputWriter:
    """Writes experiment results to timestamped log files."""
    
    def __init__(self, pattern_name: str):
        """
        Initialize the output writer.

        Args:
            pattern_name: Name of the pattern (e.g., '01_chaining')
        """
        self.pattern_name = pattern_name
        self.results_dir = settings.RESULTS_DIR

        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def write_result(
        self,
        result: Any,
        model_name: str = None,
        input_data: dict = None,
        metadata: dict = None
    ) -> Path:
        """
        Write a result to a timestamped log file.
        
        Args:
            result: The result to write (will be converted to string)
            model_name: Name of the model used (optional)
            input_data: Input data that produced this result (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Path to the created log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.pattern_name}_{timestamp}.log"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write(f"Pattern: {self.pattern_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            
            if model_name:
                f.write(f"Model: {model_name}\n")
            
            if metadata:
                f.write(f"Metadata: {json.dumps(metadata, indent=2)}\n")
            
            f.write("=" * 80 + "\n\n")
            
            # Write input if provided
            if input_data:
                f.write("--- Input ---\n")
                f.write(json.dumps(input_data, indent=2) + "\n\n")
            
            # Write result
            f.write("--- Result ---\n")
            if isinstance(result, (dict, list)):
                f.write(json.dumps(result, indent=2) + "\n")
            else:
                f.write(str(result) + "\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        return filepath
    
    def write_comparison(
        self,
        results: dict[str, Any],
        input_data: dict = None
    ) -> Path:
        """
        Write comparison results for multiple models.
        
        Args:
            results: Dictionary mapping model names to their results
            input_data: Input data that produced these results (optional)
            
        Returns:
            Path to the created log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.pattern_name}_comparison_{timestamp}.log"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write(f"Pattern: {self.pattern_name} - Model Comparison\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Models compared: {', '.join(results.keys())}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write input if provided
            if input_data:
                f.write("--- Input ---\n")
                f.write(json.dumps(input_data, indent=2) + "\n\n")
            
            # Write each model's result
            for model_name, result in results.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"{'=' * 80}\n\n")
                
                if isinstance(result, (dict, list)):
                    f.write(json.dumps(result, indent=2) + "\n")
                else:
                    f.write(str(result) + "\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        return filepath


def create_writer(pattern_name: str) -> OutputWriter:
    """
    Convenience function to create an OutputWriter.
    
    Args:
        pattern_name: Name of the pattern
        
    Returns:
        OutputWriter instance
    """
    return OutputWriter(pattern_name)