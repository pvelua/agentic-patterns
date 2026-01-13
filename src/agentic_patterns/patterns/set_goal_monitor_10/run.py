"""Goal Setting and Monitoring Pattern - Two-agent collaborative code development.

This pattern demonstrates:
1. Developer agent that creates implementation plans and generates code
2. Manager agent that monitors progress, reviews code, and provides feedback
3. Iterative improvement cycle based on manager feedback
4. Grade-based progress tracking
"""

import re
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agentic_patterns.common.model_factory import ModelFactory
from agentic_patterns.common.output_writer import create_writer
from agentic_patterns.patterns.set_goal_monitor_10.config import GoalMonitorConfig


@dataclass
class CodeIteration:
    """Represents one iteration of code development and review."""

    iteration_num: int
    implementation_plan: Optional[str] = None
    code: str = ""
    review: str = ""
    grade: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'iteration': self.iteration_num,
            'plan': self.implementation_plan,
            'code': self.code,
            'review': self.review,
            'grade': self.grade,
            'timestamp': self.timestamp
        }


class DeveloperAgent:
    """Developer agent responsible for planning and coding."""

    def __init__(self, llm, config: GoalMonitorConfig):
        self.llm = llm
        self.config = config

        # Create prompts
        self.planning_prompt = ChatPromptTemplate.from_messages([
            ("system", config.developer_system_prompt),
            ("human", config.developer_planning_template)
        ])

        self.implementation_prompt = ChatPromptTemplate.from_messages([
            ("system", config.developer_system_prompt),
            ("human", config.developer_implementation_template)
        ])

        # Create chains
        self.planning_chain = self.planning_prompt | self.llm | StrOutputParser()
        self.implementation_chain = self.implementation_prompt | self.llm | StrOutputParser()

    def create_plan(
        self,
        goal_title: str,
        goal_description: str,
        requirements: List[str],
        example_usage: str,
        verbose: bool = True
    ) -> str:
        """Create implementation plan for the goal.

        Args:
            goal_title: Title of the goal
            goal_description: Description of what needs to be built
            requirements: List of specific requirements
            example_usage: Example of how the code should be used
            verbose: Print progress

        Returns:
            Implementation plan text
        """
        if verbose:
            print("  Developer: Creating implementation plan...")

        requirements_list = "\n".join([f"{i+1}. {req}" for i, req in enumerate(requirements)])

        plan = self.planning_chain.invoke({
            'goal_title': goal_title,
            'goal_description': goal_description,
            'requirements_list': requirements_list,
            'example_usage': example_usage
        })

        if verbose:
            print(f"  Developer: Plan created ({len(plan)} characters)")

        return plan

    def implement_code(
        self,
        goal_title: str,
        requirements: List[str],
        implementation_plan: str,
        feedback: Optional[str] = None,
        current_code: Optional[str] = None,
        iteration: int = 1,
        grade: float = 0.0,
        verbose: bool = True
    ) -> str:
        """Generate Python code based on plan and optional feedback.

        Args:
            goal_title: Title of the goal
            requirements: List of requirements
            implementation_plan: Previously created plan
            feedback: Optional feedback from manager (for improvements)
            current_code: Current code to improve (if provided)
            iteration: Current iteration number
            grade: Previous grade (if feedback provided)
            verbose: Print progress

        Returns:
            Python code as string
        """
        if verbose:
            if feedback and current_code:
                print(f"  Developer: Improving code based on feedback (previous grade: {grade:.1f})...")
                print(f"  Developer: Starting from best version as baseline...")
            elif feedback:
                print(f"  Developer: Improving code based on feedback (previous grade: {grade:.1f})...")
            else:
                print("  Developer: Implementing initial version...")

        requirements_list = "\n".join([f"{i+1}. {req}" for i, req in enumerate(requirements)])

        # Prepare feedback section if provided
        if feedback and current_code:
            feedback_section = self.config.developer_improvement_section.format(
                iteration=iteration - 1,
                grade=grade,
                assessment=self._extract_assessment(feedback),
                feedback=self._extract_feedback_items(feedback)
            )
            feedback_section = f"""
CURRENT CODE (to improve):
```python
{current_code}
```

{feedback_section}"""
        elif feedback:
            feedback_section = self.config.developer_improvement_section.format(
                iteration=iteration - 1,
                grade=grade,
                assessment=self._extract_assessment(feedback),
                feedback=self._extract_feedback_items(feedback)
            )
        else:
            feedback_section = ""

        code = self.implementation_chain.invoke({
            'goal_title': goal_title,
            'requirements_list': requirements_list,
            'implementation_plan': implementation_plan,
            'feedback_section': feedback_section
        })

        # Clean up code (remove markdown if present)
        code = self._clean_code(code)

        if verbose:
            print(f"  Developer: Code generated ({len(code)} characters, {len(code.splitlines())} lines)")

        return code

    def _extract_assessment(self, review: str) -> str:
        """Extract overall assessment from review."""
        # Look for "OVERALL ASSESSMENT:" section
        match = re.search(r'OVERALL ASSESSMENT:\s*(.+?)(?=\n\n|\nFEEDBACK|$)', review, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "See full review for details."

    def _extract_feedback_items(self, review: str) -> str:
        """Extract feedback items from review."""
        # Look for "FEEDBACK FOR IMPROVEMENT:" section
        match = re.search(r'FEEDBACK FOR IMPROVEMENT:\s*(.+?)$', review, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "See full review for details."

    def _clean_code(self, code: str) -> str:
        """Remove markdown code blocks if present."""
        # Remove markdown code blocks
        code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'
        matches = re.findall(code_block_pattern, code, re.DOTALL)

        if matches:
            # If we found code blocks, use the largest one
            return max(matches, key=len).strip()

        # If no code blocks, return as-is (might already be clean code)
        return code.strip()


class ManagerAgent:
    """Manager agent responsible for code review and grading."""

    def __init__(self, llm, config: GoalMonitorConfig):
        self.llm = llm
        self.config = config

        # Create prompt
        self.review_prompt = ChatPromptTemplate.from_messages([
            ("system", config.manager_system_prompt),
            ("human", config.manager_review_template)
        ])

        # Create chain
        self.review_chain = self.review_prompt | self.llm | StrOutputParser()

    def review_code(
        self,
        goal_title: str,
        requirements: List[str],
        code: str,
        iteration: int,
        max_iterations: int,
        verbose: bool = True
    ) -> tuple[str, float]:
        """Review code and provide grade with feedback.

        Args:
            goal_title: Title of the goal
            requirements: List of requirements to check
            code: Python code to review
            iteration: Current iteration number
            max_iterations: Maximum iterations allowed
            verbose: Print progress

        Returns:
            Tuple of (review_text, grade)
        """
        if verbose:
            print(f"  Manager: Reviewing code (iteration {iteration}/{max_iterations})...")

        requirements_list = "\n".join([f"{i+1}. {req}" for i, req in enumerate(requirements)])

        review = self.review_chain.invoke({
            'goal_title': goal_title,
            'requirements_list': requirements_list,
            'code': code,
            'iteration': iteration,
            'max_iterations': max_iterations
        })

        # Extract grade from review
        grade = self.config.parse_grade(review)

        if verbose:
            print(f"  Manager: Review complete - Grade: {grade:.1f}/100")

        return review, grade


class GoalCycle:
    """Orchestrates the iterative goal-setting and monitoring cycle."""

    def __init__(
        self,
        developer_agent: DeveloperAgent,
        manager_agent: ManagerAgent,
        config: GoalMonitorConfig
    ):
        self.developer = developer_agent
        self.manager = manager_agent
        self.config = config
        self.iterations: List[CodeIteration] = []

    def run(
        self,
        goal_name: str,
        goal_spec: Dict[str, Any],
        max_iterations: int = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run the complete goal-setting and monitoring cycle.

        Args:
            goal_name: Key name of the goal from config
            goal_spec: Goal specification dictionary
            max_iterations: Max iterations (default from config)
            verbose: Print progress

        Returns:
            Results dictionary with all iterations and final outcome
        """
        if max_iterations is None:
            max_iterations = self.config.max_iterations

        if verbose:
            print(f"\n{'='*80}")
            print(f"Goal Setting and Monitoring Cycle")
            print(f"Goal: {goal_spec['title']}")
            print(f"Max Iterations: {max_iterations}")
            print(f"{'='*80}\n")

        # Step 1: Developer creates implementation plan
        if verbose:
            print("=" * 80)
            print("ITERATION 0: PLANNING")
            print("=" * 80)

        implementation_plan = self.developer.create_plan(
            goal_title=goal_spec['title'],
            goal_description=goal_spec['description'],
            requirements=goal_spec['requirements'],
            example_usage=goal_spec['example_usage'],
            verbose=verbose
        )

        # Iteration loop with best code tracking
        previous_review = None
        previous_grade = 0.0
        best_code = None
        best_grade = 0.0
        best_iteration_num = 0

        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n{'='*80}")
                print(f"ITERATION {iteration}: DEVELOP AND REVIEW")
                print(f"{'='*80}")

            # Developer implements or improves code
            # Use best code as starting point if we have it and didn't improve last time
            code_to_improve = best_code if (best_code and iteration > 1) else None

            code = self.developer.implement_code(
                goal_title=goal_spec['title'],
                requirements=goal_spec['requirements'],
                implementation_plan=implementation_plan,
                feedback=previous_review,
                current_code=code_to_improve,
                iteration=iteration,
                grade=previous_grade,
                verbose=verbose
            )

            # Manager reviews code
            review, grade = self.manager.review_code(
                goal_title=goal_spec['title'],
                requirements=goal_spec['requirements'],
                code=code,
                iteration=iteration,
                max_iterations=max_iterations,
                verbose=verbose
            )

            # Store iteration with actual code and grade
            iteration_record = CodeIteration(
                iteration_num=iteration,
                implementation_plan=implementation_plan if iteration == 1 else None,
                code=code,
                review=review,
                grade=grade
            )
            self.iterations.append(iteration_record)

            # Track best version for carry-forward
            if iteration == 1:
                # First iteration - establish baseline
                best_code = code
                best_grade = grade
                best_iteration_num = iteration
                if verbose:
                    print(f"\n  Grade: {grade:.1f}/100 (baseline)")
            elif grade > best_grade:
                # Improvement found
                best_code = code
                best_grade = grade
                best_iteration_num = iteration
                if verbose:
                    print(f"\n  Grade: {grade:.1f}/100")
                    print(f"  ✓ Grade improved: {previous_grade:.1f} → {grade:.1f}")
            else:
                # No improvement - will use best code for next iteration
                if verbose:
                    print(f"\n  Grade: {grade:.1f}/100")
                    if grade < best_grade:
                        print(f"  ⚠ Grade dropped from best: {best_grade:.1f} (iteration {best_iteration_num})")
                    else:
                        print(f"  ⚠ Grade unchanged from best: {best_grade:.1f} (iteration {best_iteration_num})")
                    print(f"  → Will reference best version for next iteration")

            # Check if we've reached passing grade
            if grade >= self.config.passing_grade:
                if verbose:
                    print(f"  ✓ Passing grade reached! ({self.config.passing_grade})")
                    print(f"\n{'='*80}")
                    print("SUCCESS: Passing grade achieved!")
                    print(f"{'='*80}\n")
                break

            # Prepare for next iteration
            # Always provide current review as feedback, but mention best version context
            if grade < best_grade:
                # Add context about best version to the review
                enhanced_review = f"""{review}

NOTE: The best version so far is from iteration {best_iteration_num} with grade {best_grade:.1f}/100.
Consider what worked well in that version while addressing the feedback above."""
                previous_review = enhanced_review
            else:
                previous_review = review

            previous_grade = grade

        # Get best iteration (highest grade)
        best_iteration = max(self.iterations, key=lambda x: x.grade)

        if verbose:
            print(f"\n{'='*80}")
            print("CYCLE COMPLETE")
            print(f"{'='*80}")
            print(f"Total Iterations: {len(self.iterations)}")
            print(f"Best Grade: {best_iteration.grade:.1f}/100 (Iteration {best_iteration.iteration_num})")
            print(f"Final Grade: {self.iterations[-1].grade:.1f}/100")

            # Show grade progression
            print(f"\nGrade Progression:")
            grades = [it.grade for it in self.iterations]
            print(f"  {' → '.join([f'{g:.1f}' for g in grades])}")

            # Calculate improvement
            improvement = best_iteration.grade - self.iterations[0].grade
            print(f"\nTotal Improvement: +{improvement:.1f} points")
            print()

        # Prepare results
        return {
            'goal_name': goal_name,
            'goal_title': goal_spec['title'],
            'total_iterations': len(self.iterations),
            'best_iteration': best_iteration.to_dict(),
            'final_iteration': self.iterations[-1].to_dict(),
            'all_iterations': [it.to_dict() for it in self.iterations],
            'grade_progression': [
                {
                    'iteration': it.iteration_num,
                    'grade': it.grade
                }
                for it in self.iterations
            ]
        }


def run(
    goal_name: str = "api_client",
    developer_model: str = None,
    manager_model: str = None,
    max_iterations: int = None,
    config: GoalMonitorConfig = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run the Goal Setting and Monitoring pattern.

    Args:
        goal_name: Name of goal from config (api_client, data_validator, task_scheduler)
        developer_model: Model for developer agent (default from config)
        manager_model: Model for manager agent (default from config)
        max_iterations: Max iterations (default from config)
        config: Configuration object (uses default if not provided)
        verbose: Print progress

    Returns:
        Dictionary with cycle results
    """
    if config is None:
        config = GoalMonitorConfig()

    if developer_model is None:
        developer_model = config.default_developer_model

    if manager_model is None:
        manager_model = config.default_manager_model

    if max_iterations is None:
        max_iterations = config.max_iterations

    # Validate goal name
    if goal_name not in config.sample_goals:
        raise ValueError(
            f"Unknown goal: {goal_name}. "
            f"Available goals: {', '.join(config.sample_goals.keys())}"
        )

    goal_spec = config.sample_goals[goal_name]

    if verbose:
        print(f"\n{'='*80}")
        print(f"Goal Setting and Monitoring Pattern")
        print(f"Developer Model: {developer_model}")
        print(f"Manager Model: {manager_model}")
        print(f"{'='*80}\n")

    # Create agents
    developer_llm = ModelFactory.create(developer_model, **config.get_developer_kwargs())
    manager_llm = ModelFactory.create(manager_model, **config.get_manager_kwargs())

    developer = DeveloperAgent(developer_llm, config)
    manager = ManagerAgent(manager_llm, config)

    # Create and run cycle
    cycle = GoalCycle(developer, manager, config)
    result = cycle.run(
        goal_name=goal_name,
        goal_spec=goal_spec,
        max_iterations=max_iterations,
        verbose=verbose
    )

    # Add model info to result
    result['developer_model'] = developer_model
    result['manager_model'] = manager_model

    return result


def compare_models(
    goal_name: str = "api_client",
    model_pairs: List[tuple[str, str]] = None,
    max_iterations: int = 4,
    config: GoalMonitorConfig = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Compare different model pairs on the same goal.

    Args:
        goal_name: Name of goal to test
        model_pairs: List of (developer_model, manager_model) tuples
        max_iterations: Max iterations for each pair
        config: Configuration object
        verbose: Print progress

    Returns:
        Comparison results
    """
    if model_pairs is None:
        model_pairs = [
            ("gpt-4o", "claude-sonnet-4-5-20250929"),
            ("gpt-4o-mini", "claude-sonnet-4-5-20250929"),
            ("claude-sonnet-4-5-20250929", "gpt-4o"),
        ]

    if config is None:
        config = GoalMonitorConfig()

    results = {}

    for dev_model, mgr_model in model_pairs:
        pair_key = f"{dev_model}__{mgr_model}"
        print(f"\n{'='*80}")
        print(f"Testing pair: Developer={dev_model}, Manager={mgr_model}")
        print(f"{'='*80}")

        try:
            result = run(
                goal_name=goal_name,
                developer_model=dev_model,
                manager_model=mgr_model,
                max_iterations=max_iterations,
                config=config,
                verbose=verbose
            )
            results[pair_key] = result
        except Exception as e:
            print(f"Error with pair ({dev_model}, {mgr_model}): {e}")
            results[pair_key] = {'error': str(e)}

    # Log comparison results
    writer = create_writer("10_set_goal_monitor")
    writer.write_comparison(
        results=results,
        metadata={
            'goal_name': goal_name,
            'max_iterations': max_iterations
        }
    )

    return results


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "compare":
            # Compare mode
            goal = sys.argv[2] if len(sys.argv) > 2 else "api_client"
            compare_models(goal_name=goal, verbose=True)
        else:
            # Single run mode
            goal = sys.argv[1] if sys.argv[1] in ["api_client", "data_validator", "task_scheduler"] else "api_client"
            dev_model = sys.argv[2] if len(sys.argv) > 2 else None
            mgr_model = sys.argv[3] if len(sys.argv) > 3 else None

            config = GoalMonitorConfig()
            result = run(
                goal_name=goal,
                developer_model=dev_model,
                manager_model=mgr_model,
                config=config,
                verbose=True
            )

            # Log results
            writer = create_writer("10_set_goal_monitor")
            writer.write_result(
                model_name=f"{result['developer_model']}__{result['manager_model']}",
                input_data={
                    'goal': goal,
                    'requirements': config.sample_goals[goal]['requirements']
                },
                result=result,
                metadata={'max_iterations': config.max_iterations}
            )
    else:
        # Default: run with api_client goal
        config = GoalMonitorConfig()
        result = run(goal_name="api_client", config=config, verbose=True)

        # Log results
        writer = create_writer("10_set_goal_monitor")
        writer.write_result(
            model_name=f"{result['developer_model']}__{result['manager_model']}",
            input_data={
                'goal': 'api_client',
                'requirements': config.sample_goals['api_client']['requirements']
            },
            result=result,
            metadata={'max_iterations': config.max_iterations}
        )
