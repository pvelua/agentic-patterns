"""Learn and Adapt Pattern - Agent Self-Improvement through Iterative Code Modification.

This pattern demonstrates an agent that improves its own code through cycles of:
1. Testing current implementation
2. Analyzing performance and failures
3. Generating improved version
4. Selecting best version for next iteration
"""

import re
import time
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agentic_patterns.common.model_factory import ModelFactory
from agentic_patterns.common.output_writer import create_writer
from agentic_patterns.patterns.learn_adapt_09.config import LearnAdaptConfig


@dataclass
class AgentVersion:
    """Represents a version of the agent code with its performance metrics."""

    version_id: int
    iteration: int
    code: str
    test_results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    overall_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'version_id': self.version_id,
            'iteration': self.iteration,
            'code': self.code,
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'overall_score': self.overall_score,
            'timestamp': self.timestamp
        }


class BenchmarkSuite:
    """Runs benchmark tests and measures performance."""

    def __init__(self, config: LearnAdaptConfig):
        self.config = config

    def run_tests(self, code: str, verbose: bool = True) -> Dict[str, Any]:
        """Run all benchmark tests on the given code.

        Args:
            code: Python function code to test
            verbose: Print test progress

        Returns:
            Dictionary with test results and performance metrics
        """
        # Execute code in isolated namespace
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return {
                'error': f'Code execution failed: {str(e)}',
                'test_results': {},
                'metrics': {
                    'success_rate': 0.0,
                    'avg_time': 0.0,
                    'complexity': self._calculate_complexity(code)
                }
            }

        if 'process_data' not in namespace:
            return {
                'error': 'Function process_data not found in code',
                'test_results': {},
                'metrics': {
                    'success_rate': 0.0,
                    'avg_time': 0.0,
                    'complexity': self._calculate_complexity(code)
                }
            }

        process_data = namespace['process_data']

        # Run each test
        test_results = {}
        execution_times = []
        total_tests = 0
        passed_tests = 0

        for test_name, test_spec in self.config.benchmark_tests.items():
            if verbose:
                print(f"  Running {test_name}...")

            result = self._run_single_test(process_data, test_name, test_spec)
            test_results[test_name] = result

            if 'subtests' in result:
                # Multiple subtests (like edge cases)
                for subtest in result['subtests']:
                    total_tests += 1
                    if subtest['passed']:
                        passed_tests += 1
                    if 'execution_time' in subtest:
                        execution_times.append(subtest['execution_time'])
            else:
                # Single test
                total_tests += 1
                if result['passed']:
                    passed_tests += 1
                if 'execution_time' in result:
                    execution_times.append(result['execution_time'])

        # Calculate metrics
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
        complexity = self._calculate_complexity(code)

        return {
            'test_results': test_results,
            'metrics': {
                'success_rate': success_rate,
                'avg_time': avg_time,
                'complexity': complexity,
                'total_tests': total_tests,
                'passed_tests': passed_tests
            }
        }

    def _run_single_test(self, func, test_name: str, test_spec: Dict) -> Dict[str, Any]:
        """Run a single test case."""
        timeout = test_spec.get('timeout', 1.0)

        # Handle tests with multiple subtests
        if 'test_cases' in test_spec:
            subtests = []
            for i, test_case in enumerate(test_spec['test_cases']):
                subtest_result = self._execute_test(
                    func,
                    test_case['input'],
                    test_case.get('expected'),
                    timeout
                )
                subtest_result['subtest_id'] = i
                subtests.append(subtest_result)

            return {
                'description': test_spec['description'],
                'subtests': subtests,
                'passed': all(st['passed'] for st in subtests)
            }

        # Single test case
        return self._execute_test(
            func,
            test_spec['input'],
            test_spec.get('expected'),
            timeout,
            test_spec.get('expected_keys')
        )

    def _execute_test(
        self,
        func,
        test_input: Any,
        expected: Any = None,
        timeout: float = 1.0,
        expected_keys: List[str] = None
    ) -> Dict[str, Any]:
        """Execute a test and measure performance."""
        try:
            start_time = time.time()
            result = func(test_input)
            execution_time = time.time() - start_time

            # Check timeout
            if execution_time > timeout:
                return {
                    'passed': False,
                    'error': f'Timeout: {execution_time:.4f}s > {timeout}s',
                    'execution_time': execution_time,
                    'result': result
                }

            # Check result
            if expected is not None:
                if expected == 'empty_list_handled':
                    # Special case: just check it doesn't crash and returns something
                    passed = result is not None
                elif expected == 'none_handled':
                    # Special case: check None is handled gracefully
                    passed = result is not None
                else:
                    passed = result == expected
            elif expected_keys is not None:
                # Check that result has expected keys
                passed = (
                    isinstance(result, dict) and
                    all(key in result for key in expected_keys)
                )
            else:
                # No validation, just check it runs
                passed = True

            return {
                'passed': passed,
                'execution_time': execution_time,
                'result': result,
                'expected': expected
            }

        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'execution_time': 0.0
            }

    def _calculate_complexity(self, code: str) -> int:
        """Calculate code complexity (simplified as non-empty lines)."""
        lines = [line.strip() for line in code.split('\n')]
        non_empty_lines = [line for line in lines if line and not line.startswith('#')]
        return len(non_empty_lines)


class VersionArchive:
    """Manages archive of agent versions and selects best performers."""

    def __init__(self, config: LearnAdaptConfig):
        self.config = config
        self.versions: List[AgentVersion] = []
        self._next_version_id = 0

    def add_version(self, version: AgentVersion):
        """Add a version to the archive."""
        self.versions.append(version)

    def get_best_version(self) -> Optional[AgentVersion]:
        """Get the version with highest performance score."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.overall_score)

    def get_version_history_summary(self, max_versions: int = 3) -> str:
        """Get formatted summary of recent versions for LLM context."""
        if not self.versions:
            return "No previous versions."

        # Get recent versions (up to max_versions)
        recent = self.versions[-max_versions:]

        summary_parts = []
        for v in recent:
            metrics = v.performance_metrics
            summary_parts.append(
                f"Version {v.version_id} (Iteration {v.iteration}):\n"
                f"  - Success Rate: {metrics['success_rate']:.1%}\n"
                f"  - Avg Time: {metrics['avg_time']:.4f}s\n"
                f"  - Complexity: {metrics['complexity']} lines\n"
                f"  - Overall Score: {v.overall_score:.3f}"
            )

        return "\n\n".join(summary_parts)

    def calculate_score_for_metrics(
        self,
        success_rate: float,
        avg_time: float,
        complexity: int
    ) -> float:
        """Calculate score using observed max values."""
        max_time = max((v.performance_metrics['avg_time'] for v in self.versions), default=avg_time)
        max_complexity = max((v.performance_metrics['complexity'] for v in self.versions), default=complexity)

        # Add current values to max calculation
        max_time = max(max_time, avg_time)
        max_complexity = max(max_complexity, complexity)

        return self.config.calculate_score(
            success_rate=success_rate,
            avg_time=avg_time,
            max_time=max_time,
            complexity=complexity,
            max_complexity=max_complexity
        )


class AdaptationEngine:
    """Uses LLM to analyze performance and generate improved code."""

    def __init__(self, llm, config: LearnAdaptConfig):
        self.llm = llm
        self.config = config

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", config.adaptation_system_prompt),
            ("human", config.improvement_prompt_template)
        ])

        # Create chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_improvement(
        self,
        current_code: str,
        test_results: Dict[str, Any],
        metrics: Dict[str, float],
        version_history: str,
        verbose: bool = True
    ) -> str:
        """Generate improved version of code based on test results.

        Args:
            current_code: Current implementation
            test_results: Results from benchmark tests
            metrics: Performance metrics
            version_history: Summary of past versions
            verbose: Print generation progress

        Returns:
            Improved code
        """
        # Format test results for LLM
        test_summary = self._format_test_results(test_results)
        failed_tests = self._format_failed_tests(test_results)

        if verbose:
            print(f"  Generating improved version...")
            print(f"    Success rate: {metrics['success_rate']:.1%}")
            print(f"    Failed tests: {failed_tests[:100]}...")

        # Generate improvement
        try:
            improved_code = self.chain.invoke({
                'current_code': current_code,
                'test_results': test_summary,
                'success_rate': metrics['success_rate'],
                'avg_time': metrics['avg_time'],
                'complexity': metrics['complexity'],
                'failed_tests': failed_tests if failed_tests else "None - all tests passing!",
                'version_history': version_history
            })

            # Extract Python code from response
            improved_code = self._extract_code(improved_code)

            return improved_code

        except Exception as e:
            print(f"  Error generating improvement: {e}")
            return current_code

    def _format_test_results(self, test_results: Dict[str, Any]) -> str:
        """Format test results for LLM consumption."""
        lines = []
        for test_name, result in test_results.items():
            if 'subtests' in result:
                passed_count = sum(1 for st in result['subtests'] if st['passed'])
                total_count = len(result['subtests'])
                lines.append(f"- {test_name}: {passed_count}/{total_count} subtests passed")
            else:
                status = "PASS" if result['passed'] else "FAIL"
                lines.append(f"- {test_name}: {status}")

        return "\n".join(lines)

    def _format_failed_tests(self, test_results: Dict[str, Any]) -> str:
        """Format details of failed tests for LLM."""
        failed = []

        for test_name, result in test_results.items():
            if 'subtests' in result:
                for subtest in result['subtests']:
                    if not subtest['passed']:
                        error = subtest.get('error', 'Assertion failed')
                        expected = subtest.get('expected')
                        actual = subtest.get('result')
                        failed.append(
                            f"{test_name} (subtest {subtest['subtest_id']}): {error}\n"
                            f"  Expected: {expected}\n"
                            f"  Got: {actual}"
                        )
            elif not result['passed']:
                error = result.get('error', 'Assertion failed')
                expected = result.get('expected')
                actual = result.get('result')
                failed.append(
                    f"{test_name}: {error}\n"
                    f"  Expected: {expected}\n"
                    f"  Got: {actual}"
                )

        return "\n\n".join(failed) if failed else ""

    def _extract_code(self, llm_response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to extract code from markdown code blocks
        code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

        if matches:
            # Return the largest code block (likely the function)
            return max(matches, key=len).strip()

        # If no code blocks, look for function definition
        if 'def process_data' in llm_response:
            # Try to extract from first 'def' to end or next '```'
            lines = llm_response.split('\n')
            code_lines = []
            in_function = False

            for line in lines:
                if 'def process_data' in line:
                    in_function = True

                if in_function:
                    if line.strip().startswith('```') or line.strip().startswith('def ') and code_lines:
                        break
                    code_lines.append(line)

            if code_lines:
                return '\n'.join(code_lines).strip()

        # Last resort: return the whole response
        return llm_response.strip()


def run(
    model_name: str = "gpt-4o",
    max_iterations: int = None,
    config: LearnAdaptConfig = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run the Learn and Adapt pattern with agent self-improvement.

    Args:
        model_name: Name of the LLM model to use
        max_iterations: Maximum number of improvement iterations (default from config)
        config: Configuration object (uses default if not provided)
        verbose: Print progress information

    Returns:
        Dictionary with results including version history and best version
    """
    if config is None:
        config = LearnAdaptConfig()

    if max_iterations is None:
        max_iterations = config.max_iterations

    if verbose:
        print(f"\n{'='*80}")
        print(f"Learn and Adapt Pattern - Agent Self-Improvement")
        print(f"Model: {model_name}")
        print(f"Max Iterations: {max_iterations}")
        print(f"{'='*80}\n")

    # Create LLM
    llm = ModelFactory.create(model_name, **config.get_model_kwargs())

    # Initialize components
    benchmark = BenchmarkSuite(config)
    archive = VersionArchive(config)
    adaptation_engine = AdaptationEngine(llm, config)

    # Start with initial code
    current_code = config.initial_agent_code

    # Main improvement loop
    for iteration in range(max_iterations):
        if verbose:
            print(f"\nIteration {iteration + 1}/{max_iterations}")
            print(f"{'-'*80}")

        # Run benchmarks on current code
        if verbose:
            print("Running benchmark tests...")
        test_run = benchmark.run_tests(current_code, verbose=verbose)

        if 'error' in test_run:
            if verbose:
                print(f"  Error: {test_run['error']}")
            metrics = test_run['metrics']
            test_results = {}
        else:
            test_results = test_run['test_results']
            metrics = test_run['metrics']

        # Calculate overall score
        score = archive.calculate_score_for_metrics(
            success_rate=metrics['success_rate'],
            avg_time=metrics['avg_time'],
            complexity=metrics['complexity']
        )

        # Create version record
        version = AgentVersion(
            version_id=archive._next_version_id,
            iteration=iteration,
            code=current_code,
            test_results=test_results,
            performance_metrics=metrics,
            overall_score=score
        )
        archive.add_version(version)
        archive._next_version_id += 1

        if verbose:
            print(f"\nResults:")
            print(f"  Success Rate: {metrics['success_rate']:.1%}")
            print(f"  Avg Execution Time: {metrics['avg_time']:.4f}s")
            print(f"  Code Complexity: {metrics['complexity']} lines")
            print(f"  Overall Score: {score:.3f}")

        # Check if we've reached the threshold
        if metrics['success_rate'] >= config.performance_threshold:
            if verbose:
                print(f"\n✓ Performance threshold reached! ({config.performance_threshold:.1%})")
            break

        # Generate improved version for next iteration
        if iteration < max_iterations - 1:  # Don't improve on last iteration
            version_history = archive.get_version_history_summary()
            improved_code = adaptation_engine.generate_improvement(
                current_code=current_code,
                test_results=test_results,
                metrics=metrics,
                version_history=version_history,
                verbose=verbose
            )
            current_code = improved_code

    # Get best version
    best_version = archive.get_best_version()

    if verbose:
        print(f"\n{'='*80}")
        print(f"Self-Improvement Complete!")
        print(f"{'='*80}")
        print(f"Best Version: {best_version.version_id} (Iteration {best_version.iteration})")
        print(f"  Success Rate: {best_version.performance_metrics['success_rate']:.1%}")
        print(f"  Overall Score: {best_version.overall_score:.3f}")
        print()

    # Prepare results
    return {
        'model': model_name,
        'total_iterations': len(archive.versions),
        'best_version': best_version.to_dict(),
        'all_versions': [v.to_dict() for v in archive.versions],
        'improvement_trajectory': [
            {
                'iteration': v.iteration,
                'score': v.overall_score,
                'success_rate': v.performance_metrics['success_rate']
            }
            for v in archive.versions
        ]
    }


def compare_models(
    model_names: List[str] = None,
    max_iterations: int = 5,
    config: LearnAdaptConfig = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Compare multiple models on the Learn and Adapt pattern.

    Args:
        model_names: List of model names to compare
        max_iterations: Maximum iterations for each model
        config: Configuration object
        verbose: Print progress

    Returns:
        Comparison results across models
    """
    if model_names is None:
        model_names = ["gpt-4o-mini", "claude-sonnet-4-5-20250929"]

    if config is None:
        config = LearnAdaptConfig()

    results = {}

    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Testing model: {model_name}")
        print(f"{'='*80}")

        try:
            result = run(
                model_name=model_name,
                max_iterations=max_iterations,
                config=config,
                verbose=verbose
            )
            results[model_name] = result
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    # Log comparison results
    writer = create_writer("09_learn_adapt")
    writer.write_comparison(
        results=results,
        metadata={
            'max_iterations': max_iterations,
            'performance_threshold': config.performance_threshold
        }
    )

    return results


if __name__ == "__main__":
    # Get model name from command line or use default
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o"

    # Check if compare mode
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_models(verbose=True)
    else:
        # Run single model
        config = LearnAdaptConfig()
        result = run(model_name=model_name, config=config, verbose=True)

        # Log results
        writer = create_writer("09_learn_adapt")
        writer.write_result(
            model_name=model_name,
            input_data={'initial_code': config.initial_agent_code},
            result=result,
            metadata={'max_iterations': config.max_iterations}
        )
