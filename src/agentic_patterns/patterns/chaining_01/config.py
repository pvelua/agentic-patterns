"""Configuration for the chaining pattern."""
from dataclasses import dataclass


@dataclass
class ChainingConfig:
    """Configuration for the chaining pattern."""
    
    # Model parameters
    temperature: float = 0.0
    max_tokens: int = 2000
    
    # Prompts
    extraction_prompt: str = (
        "Extract the technical specifications from the following text:\n\n{text_input}"
    )
    
    transformation_prompt: str = (
        "Transform the following specifications into a JSON object with "
        "'cpu', 'memory', and 'storage' as keys:\n\n{specifications}"
    )
    
    def get_model_kwargs(self) -> dict:
        """Get kwargs for model initialization."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }