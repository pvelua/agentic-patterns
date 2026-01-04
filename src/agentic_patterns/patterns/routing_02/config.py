"""Configuration for the Routing pattern."""

from dataclasses import dataclass


@dataclass
class RoutingConfig:
    """Configuration for routing pattern with coordinator and handlers."""

    # LLM parameters
    temperature: float = 0
    max_tokens: int = 2000

    # Router prompt - determines which handler to delegate to
    router_prompt: str = """Analyze the user's request and determine which specialist handler should process it.
    - If the request is related to booking flights or hotels, output 'booker'.
    - For all other general information questions, output 'info'.
    - If the request is unclear or doesn't fit either category, output 'unclear'.

ONLY output one word: 'booker', 'info', or 'unclear'."""

    def get_model_kwargs(self) -> dict:
        """Return parameters for ModelFactory.create()"""
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
