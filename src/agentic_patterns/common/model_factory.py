from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from .config import settings

class ModelFactory:
    """Factory for creating LangChain model instances"""
    
    @staticmethod
    def create(model_name: str, **kwargs):
        """
        Create a LangChain model instance
        
        Args:
            model_name: Model identifier (e.g., 'gpt-4', 'claude-sonnet-4')
            **kwargs: Additional model parameters
        """
        defaults = {
            'temperature': 0.7,
            'max_tokens': 2000,
        }
        params = {**defaults, **kwargs}
        
        if model_name.startswith('gpt'):
            return ChatOpenAI(
                model=model_name,
                api_key=settings.OPENAI_API_KEY,
                **params
            )
        elif model_name.startswith('claude'):
            return ChatAnthropic(
                model=model_name,
                api_key=settings.ANTHROPIC_API_KEY,
                **params
            )
        elif model_name.startswith('gemini'):
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=settings.GOOGLE_API_KEY,
                **params
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")