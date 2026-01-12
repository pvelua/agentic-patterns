"""Configuration for the Learn and Adapt pattern - Agent Self-Improvement."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class LearnAdaptConfig:
    """Configuration for agent self-improvement pattern."""

    # LLM parameters
    temperature: float = 0.3  # Balanced for code generation
    max_tokens: int = 4000

    # Self-improvement parameters
    max_iterations: int = 5
    performance_threshold: float = 0.95  # Stop if we reach this score

    # Scoring weights (must sum to 1.0)
    weight_success: float = 0.5  # Success rate weight
    weight_time: float = 0.3  # Execution time weight (lower is better)
    weight_complexity: float = 0.2  # Code complexity weight (lower is better)

    # Initial agent code template
    initial_agent_code: str = '''def process_data(data):
    """Process input data and return results.

    Args:
        data: Input data (list, dict, or string)

    Returns:
        Processed results based on data type
    """
    if isinstance(data, list):
        # Basic numeric processing
        if all(isinstance(x, (int, float)) for x in data):
            return {
                'sum': sum(data),
                'average': sum(data) / len(data),
                'max': max(data)
            }
    elif isinstance(data, str):
        # Basic string processing
        words = data.split()
        return {
            'word_count': len(words),
            'char_count': len(data)
        }
    return None
'''

    # System prompt for the adaptation engine
    adaptation_system_prompt: str = """You are an expert code improvement agent specializing in Python optimization.

Your task is to analyze the current code implementation and its test results, then generate an improved version.

Key objectives:
1. Fix any failing tests by addressing the root causes
2. Improve execution speed where possible
3. Keep code simple and readable (avoid unnecessary complexity)
4. Handle edge cases gracefully (empty inputs, None values, invalid types)
5. Maintain backward compatibility with the function signature

When generating improved code:
- Provide ONLY the complete improved function definition
- Do NOT include explanations, comments about changes, or example usage
- Ensure the function signature remains: def process_data(data):
- Focus on correctness first, then performance
- Add proper error handling and input validation

Your response should contain ONLY the Python function code, nothing else."""

    # Prompt template for code improvement
    improvement_prompt_template: str = """Analyze the following code and its test results, then provide an improved version.

CURRENT CODE:
```python
{current_code}
```

TEST RESULTS:
{test_results}

PERFORMANCE METRICS:
- Success Rate: {success_rate:.1%}
- Average Execution Time: {avg_time:.4f}s
- Code Complexity: {complexity} lines

FAILED TESTS:
{failed_tests}

ARCHIVE OF PAST VERSIONS:
{version_history}

Based on this analysis, generate an improved version of the process_data function that:
1. Fixes all failing tests
2. Improves performance where possible
3. Maintains code simplicity

Provide ONLY the improved function code."""

    # Benchmark test definitions
    benchmark_tests: Dict[str, Dict] = field(default_factory=lambda: {
        'test_numeric_processing': {
            'description': 'Process numeric list and compute statistics',
            'input': [10, 20, 30, 40, 50],
            'expected': {
                'sum': 150,
                'average': 30.0,
                'max': 50,
                'min': 10,
                'count': 5
            },
            'timeout': 1.0
        },
        'test_edge_cases': {
            'description': 'Handle edge cases: empty list, single item, None',
            'test_cases': [
                {'input': [], 'expected': 'empty_list_handled'},
                {'input': [42], 'expected': {'sum': 42, 'average': 42.0, 'max': 42, 'min': 42, 'count': 1}},
                {'input': None, 'expected': 'none_handled'},
            ],
            'timeout': 1.0
        },
        'test_string_operations': {
            'description': 'Count word frequencies in text',
            'input': 'the quick brown fox jumps over the lazy dog',
            'expected': {
                'word_count': 9,
                'char_count': 43,
                'unique_words': 8,
                'word_freq': {'the': 2, 'quick': 1, 'brown': 1, 'fox': 1, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1}
            },
            'timeout': 1.0
        },
        'test_performance': {
            'description': 'Process large dataset efficiently',
            'input': list(range(10000)),
            'expected_keys': ['sum', 'average', 'max', 'min', 'count'],
            'timeout': 0.5  # Must complete in 500ms
        }
    })

    def get_model_kwargs(self) -> dict:
        """Return parameters for ModelFactory.create()"""
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

    def calculate_score(
        self,
        success_rate: float,
        avg_time: float,
        max_time: float,
        complexity: int,
        max_complexity: int
    ) -> float:
        """Calculate weighted performance score.

        Args:
            success_rate: Test success rate (0.0 to 1.0)
            avg_time: Average execution time in seconds
            max_time: Maximum time observed across all versions
            complexity: Code complexity (lines of code)
            max_complexity: Maximum complexity observed

        Returns:
            Weighted score (0.0 to 1.0), higher is better
        """
        # Normalize time (0 = worst, 1 = best)
        if max_time > 0:
            time_score = 1.0 - (avg_time / max_time)
        else:
            time_score = 1.0

        # Normalize complexity (0 = worst, 1 = best)
        if max_complexity > 0:
            complexity_score = 1.0 - (complexity / max_complexity)
        else:
            complexity_score = 1.0

        # Calculate weighted score
        score = (
            self.weight_success * success_rate +
            self.weight_time * time_score +
            self.weight_complexity * complexity_score
        )

        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
