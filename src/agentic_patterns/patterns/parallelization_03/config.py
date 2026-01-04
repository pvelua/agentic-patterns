"""Configuration for the Parallelization pattern."""

from dataclasses import dataclass


@dataclass
class ParallelizationConfig:
    """Configuration for parallelization pattern with multiple concurrent chains."""

    # LLM parameters
    temperature: float = 0.7
    max_tokens: int = 2000

    # Prompts for parallel chains
    summarize_prompt: str = "Summarize the following topic concisely:"
    questions_prompt: str = "Generate three interesting questions about the following topic:"
    terms_prompt: str = "Identify 5-10 key terms from the following topic, separated by commas:"

    # Synthesis prompt - combines results from parallel chains
    synthesis_prompt: str = """Based on the following information:
    Summary: {summary}
    Related Questions: {questions}
    Key Terms: {key_terms}

    Synthesize a comprehensive answer."""

    def get_model_kwargs(self) -> dict:
        """Return parameters for ModelFactory.create()"""
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
