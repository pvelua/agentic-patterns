"""Configuration for the Planning pattern."""

from dataclasses import dataclass


@dataclass
class PlanningConfig:
    """Configuration for planning pattern with multi-step plan generation and execution."""

    # LLM parameters for planner
    planner_temperature: float = 0.3  # Slightly higher for creative planning
    planner_max_tokens: int = 2000

    # LLM parameters for executor
    executor_temperature: float = 0.2  # Lower for focused execution
    executor_max_tokens: int = 3000

    # Planning parameters
    max_plan_steps: int = 8  # Maximum number of steps in a plan
    min_plan_steps: int = 3  # Minimum number of steps in a plan

    # Planner system prompt
    planner_system_prompt: str = """You are an expert strategic planner and task decomposer.
Your role is to analyze complex tasks and break them down into clear, actionable steps.

When creating a plan:
1. Analyze the task thoroughly to understand its requirements and objectives
2. Break down the task into logical, sequential steps
3. Each step should be specific, actionable, and contribute to the final goal
4. Number each step clearly (1., 2., 3., etc.)
5. Keep steps focused and manageable
6. Ensure the steps flow logically from one to the next
7. Include a brief rationale for why this approach will work

Output your plan in this exact format:
## Plan for: [Task Title]

### Rationale
[Brief explanation of the overall approach]

### Steps
1. [First step - be specific]
2. [Second step - be specific]
3. [Third step - be specific]
...

### Expected Outcome
[What the final result should look like]"""

    # Executor system prompt
    executor_system_prompt: str = """You are an expert task executor who follows plans precisely.
Your role is to execute each step of a given plan thoroughly and effectively.

When executing a plan:
1. Follow each step in the exact order specified
2. Be thorough and detailed in your execution
3. Build upon the results of previous steps
4. Stay focused on the plan's objectives
5. Provide clear, well-structured output

Your execution should be comprehensive and directly address each step of the plan."""

    def get_planner_kwargs(self) -> dict:
        """Return parameters for planner ModelFactory.create()"""
        return {
            'temperature': self.planner_temperature,
            'max_tokens': self.planner_max_tokens
        }

    def get_executor_kwargs(self) -> dict:
        """Return parameters for executor ModelFactory.create()"""
        return {
            'temperature': self.executor_temperature,
            'max_tokens': self.executor_max_tokens
        }

    def get_model_kwargs(self) -> dict:
        """Return parameters for ModelFactory.create() (backward compatibility)"""
        return self.get_planner_kwargs()
