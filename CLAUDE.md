# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Experimental framework for testing various agentic design patterns using LangChain and LangGraph. The project implements multiple LLM agent patterns (chaining, routing, parallelization, reflection, tool use) and allows comparison across different models (OpenAI GPT, Anthropic Claude, Google Gemini).

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Configure API keys
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
```

### Running Patterns
```bash
# Run a specific pattern with default model (gpt-4)
uv run src/agentic_patterns/patterns/chaining_01/run.py

# Run with a specific model
uv run src/agentic_patterns/patterns/chaining_01/run.py claude-sonnet-4-5-20250929
uv run src/agentic_patterns/patterns/chaining_01/run.py gpt-4o
uv run src/agentic_patterns/patterns/chaining_01/run.py gemini-2.5-flash
```

### Code Quality
```bash
# Lint code
uv run ruff check src/

# Run tests
uv run pytest
```

### Package Management
```bash
# Add new dependency
uv add <package-name>

# Add development dependency
uv add --dev <package-name>
```

## Architecture

### Core Components

**ModelFactory** (src/agentic_patterns/common/model_factory.py)
- Central factory for creating LangChain LLM instances
- Supports OpenAI (gpt-*), Anthropic (claude-*), Google (gemini-*) models
- Automatically selects the correct LangChain class based on model name prefix
- Handles API key injection from environment settings
- Default params: temperature=0.7, max_tokens=2000

**Settings/Config** (src/agentic_patterns/common/config.py)
- Uses pydantic_settings with .env file loading
- Automatically finds project root and loads .env from there
- Required: OPENAI_API_KEY, ANTHROPIC_API_KEY
- Optional: GOOGLE_API_KEY, LOG_LEVEL, RESULTS_DIR
- All pattern code should import from this central settings

**OutputWriter** (src/agentic_patterns/common/output_writer.py)
- Standardized logging for all pattern results
- Writes to experiments/results/ with timestamps
- Two methods:
  - write_result(): Single model execution
  - write_comparison(): Multi-model comparison
- Auto-creates directory structure, includes metadata and input data

### Pattern Structure

Each pattern lives in `src/agentic_patterns/patterns/<pattern_name>/` and follows this structure:

1. **config.py**: Pattern-specific configuration as a dataclass
   - Should include model parameters (temperature, max_tokens)
   - Should include pattern-specific prompts/settings
   - Must have get_model_kwargs() method returning dict for ModelFactory
   - For patterns with multiple roles (e.g., reflection), may include separate model configurations

2. **run.py**: Main implementation with two key functions
   - run(): Execute pattern with single model (or multiple models for multi-role patterns), returns result
   - compare_models(): Run pattern across multiple models for comparison
   - Both functions should use OutputWriter for logging
   - Main block should accept model name as CLI argument

3. **__init__.py**: Export public API (run, compare_models functions)

### Pattern-Specific Variations

**Reflection Pattern (reflection_04)**: Supports dual-model approach with separate creator and critic roles

- **Configuration**: Includes separate settings for creator and critic models
  ```python
  default_creator_model: str = "gpt-4"
  default_critic_model: str = "claude-sonnet-4-5-20250929"
  creator_temperature: float = 0.1  # Deterministic code generation
  critic_temperature: float = 0.3   # More diverse critique
  ```
- **Methods**: `get_creator_kwargs()` and `get_critic_kwargs()` for separate model configurations
- **Function Signature**: `run(creator_model=None, critic_model=None, task_prompt=None, config=None, verbose=True)`
- **Two LLM Instances**: Creates separate `creator_llm` and `critic_llm` instances
- **Usage Examples**:
  ```bash
  # Use defaults (gpt-4 creator, claude-sonnet-4-5-20250929 critic)
  uv run src/agentic_patterns/patterns/reflection_04/run.py

  # Specify both models
  uv run src/agentic_patterns/patterns/reflection_04/run.py gpt-4.1-2025-04-14 claude-sonnet-4-5-20251101

  # Programmatic usage
  uv run python -c "from agentic_patterns.patterns.reflection_04 import run; run('gpt-4o', 'claude-sonnet-4-5-20250929', 'Your task')"
  ```
- **Compare Models**: Accepts list of (creator, critic) tuples
  ```python
  compare_models(model_pairs=[
      ("gpt-4", "claude-sonnet-4-5-20250929"),
      ("gpt-4o", "claude-sonnet-4-5-20250929"),
      ("claude-sonnet-4-5-20250929", "gpt-4o"),
  ])
  ```
- **Benefits**: Leverages different model strengths (e.g., GPT for generation, Claude for critical analysis)

### LangChain Patterns

The codebase uses LangChain Expression Language (LCEL) for composing chains:

```python
# Example from chaining_01
chain = prompt | llm | StrOutputParser()

# Multi-step with variable passing
full_chain = (
    {"specifications": extraction_chain}
    | prompt_transform
    | llm
    | StrOutputParser()
)
```

Key LangChain imports commonly used:
- `langchain_core.prompts.ChatPromptTemplate`
- `langchain_core.output_parsers.StrOutputParser`
- Model-specific: `langchain_openai.ChatOpenAI`, `langchain_anthropic.ChatAnthropic`, `langchain_google_genai.ChatGoogleGenerativeAI`

### Results and Logging

All pattern execution results are automatically logged to `experiments/results/`:
- Filename format: `{pattern_name}_{timestamp}.log` or `{pattern_name}_comparison_{timestamp}.log`
- Directory is auto-created if it doesn't exist
- Logs include: timestamp, model name, input data, metadata, and full results
- Use create_writer() helper function from common module

## Model Naming Conventions

When adding support for new models:
- OpenAI models must start with 'gpt' prefix
- Anthropic models must start with 'claude' prefix
- Google models must start with 'gemini' prefix
- ModelFactory uses string.startswith() to route to correct provider

Supported models (as of 2025):
- **OpenAI**: gpt-4.1-2025-04-14, gpt-4.1-mini-2025-04-14, gpt-4o, gpt-4o-mini, gpt-4-turbo
- **Anthropic**: claude-sonnet-4-5-20250929, claude-opus-4-5-20251101, claude-3-5-haiku-20241022
- **Google**: gemini-3-pro-preview, gemini-3-flash-preview, gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite

Note: Deprecated models include gpt-3.5-turbo (OpenAI), gemini-pro, and gemini-1.5-pro (Google, retired April 2025)

## Adding New Patterns

When implementing a new pattern:

1. Create directory: `src/agentic_patterns/patterns/<pattern_name>/`
2. Add three files: `__init__.py`, `config.py`, `run.py`
3. Use ModelFactory.create() for model instantiation
4. Use create_writer() from common.output_writer for logging
5. Implement both run() and compare_models() functions
6. Follow LCEL composition style for chains
7. Include CLI support in `if __name__ == "__main__"` block
8. Pattern config should be a dataclass with get_model_kwargs() method
9. For patterns with multiple roles (e.g., creator/critic), consider:
   - Separate model configuration for each role (default_creator_model, default_critic_model)
   - Separate temperature/parameter settings per role
   - Multiple get_*_kwargs() methods (get_creator_kwargs(), get_critic_kwargs())
   - Function signatures accepting multiple model parameters
   - Enhanced logging showing which model is used for which role
