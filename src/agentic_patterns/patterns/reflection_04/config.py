"""Configuration for the Reflection pattern."""

from dataclasses import dataclass


@dataclass
class ReflectionConfig:
    """Configuration for reflection pattern with iterative self-improvement."""

    # Default model names
    default_creator_model: str = "gpt-4"
    default_critic_model: str = "claude-sonnet-4-5-20250929"

    # Creator LLM parameters
    creator_temperature: float = 0.1  # Lower temperature for more deterministic outputs
    creator_max_tokens: int = 2000

    # Critic LLM parameters
    critic_temperature: float = 0.3  # Slightly higher for more diverse critique
    critic_max_tokens: int = 2000

    # Reflection loop parameters
    max_iterations: int = 3
    stopping_phrase: str = "CODE_IS_PERFECT"

    # Default task prompt (can be overridden)
    task_prompt: str = """Your task is to create a Python function named `calculate_factorial`.
This function should do the following:
1. Accept a single integer `n` as input.
2. Calculate its factorial (n!).
3. Include a clear docstring explaining what the function does.
4. Handle edge cases: The factorial of 0 is 1.
5. Handle invalid input: Raise a ValueError if the input is a negative number."""

    # Reflector system prompt - defines the critic role
    reflector_system_prompt: str = """You are a senior software engineer and an expert in Python.
Your role is to perform a meticulous code review.
Critically evaluate the provided Python code based on the original task requirements.
Look for bugs, style issues, missing edge cases, and areas for improvement.
If the code is perfect and meets all requirements, respond with the single phrase 'CODE_IS_PERFECT'.
Otherwise, provide a bulleted list of your critiques."""

    def get_creator_kwargs(self) -> dict:
        """Return parameters for creator ModelFactory.create()"""
        return {
            'temperature': self.creator_temperature,
            'max_tokens': self.creator_max_tokens
        }

    def get_critic_kwargs(self) -> dict:
        """Return parameters for critic ModelFactory.create()"""
        return {
            'temperature': self.critic_temperature,
            'max_tokens': self.critic_max_tokens
        }

    def get_model_kwargs(self) -> dict:
        """Return parameters for ModelFactory.create() (backward compatibility)"""
        return self.get_creator_kwargs()
