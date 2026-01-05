"""Configuration for the Tool Use pattern."""

from dataclasses import dataclass


@dataclass
class ToolUseConfig:
    """Configuration for tool use pattern with function calling capabilities."""

    # LLM parameters
    temperature: float = 0.0  # Lower temperature for more deterministic tool selection
    max_tokens: int = 2000

    # System prompt for the agent
    system_prompt: str = """You are a helpful research assistant with access to specialized search tools.
Use the available tools to answer user questions accurately.
Always cite which tool you used to find the information."""

    # Simulated knowledge base for different search tools
    tech_search_results: dict = None
    science_search_results: dict = None
    history_search_results: dict = None

    def __post_init__(self):
        """Initialize simulated search results if not provided."""
        if self.tech_search_results is None:
            self.tech_search_results = {
                "python": "Python is a high-level programming language known for its simplicity and readability. Created by Guido van Rossum in 1991, it supports multiple programming paradigms including object-oriented, functional, and procedural programming.",
                "rust": "Rust is a systems programming language focused on safety, concurrency, and performance. It prevents memory errors through its ownership system and has no garbage collector.",
                "quantum computing": "Quantum computing leverages quantum-mechanical phenomena like superposition and entanglement to perform calculations. Quantum computers use qubits instead of classical bits, enabling exponential speedup for certain problems.",
                "machine learning": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming. It uses statistical techniques to give computers the ability to learn from data.",
                "blockchain": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records called blocks. Each block contains a cryptographic hash of the previous block, timestamp, and transaction data.",
                "default": "No specific information found in the technology database. This appears to be a technology-related query that requires further research."
            }

        if self.science_search_results is None:
            self.science_search_results = {
                "photosynthesis": "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose. It occurs in chloroplasts and involves light-dependent reactions and the Calvin cycle.",
                "dna": "DNA (deoxyribonucleic acid) is the hereditary material in humans and most organisms. Its double helix structure was discovered by Watson and Crick in 1953. DNA contains genetic instructions for development, functioning, and reproduction.",
                "black holes": "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse at the end of their life cycle.",
                "crispr": "CRISPR (Clustered Regularly Interspaced Short Palindromic Repeats) is a revolutionary gene-editing technology that allows scientists to precisely modify DNA sequences and gene function.",
                "climate change": "Climate change refers to long-term shifts in global temperatures and weather patterns. Since the 1800s, human activities, particularly burning fossil fuels, have been the main driver of climate change.",
                "default": "No specific information found in the science database. This appears to be a science-related query that requires additional research."
            }

        if self.history_search_results is None:
            self.history_search_results = {
                "roman empire": "The Roman Empire was one of the largest empires in ancient history, lasting from 27 BCE to 476 CE in the West. At its peak, it controlled territories across Europe, North Africa, and the Middle East.",
                "world war ii": "World War II (1939-1945) was a global conflict involving most of the world's nations. It resulted in an estimated 70-85 million fatalities and led to significant geopolitical changes including the emergence of the United States and Soviet Union as superpowers.",
                "industrial revolution": "The Industrial Revolution was a period of major industrialization from the late 18th to early 19th century. It began in Britain and spread worldwide, transforming economies from agrarian to industrial.",
                "renaissance": "The Renaissance was a period of cultural, artistic, and intellectual rebirth in Europe from the 14th to 17th century. It saw revival of classical learning and marked the transition from the Middle Ages to modernity.",
                "ancient egypt": "Ancient Egypt was a civilization in northeastern Africa lasting from around 3100 BCE to 30 BCE. Known for pyramids, hieroglyphics, and pharaohs, it made significant contributions to mathematics, medicine, and architecture.",
                "default": "No specific information found in the history database. This appears to be a history-related query that needs more specialized sources."
            }

    def get_model_kwargs(self) -> dict:
        """Return parameters for ModelFactory.create()"""
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
