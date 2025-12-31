# Agentic Patterns

Experimental framework for testing various agentic design patterns using LangChain and LangGraph.

## Setup

1. **Install UV** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone/navigate to the project**:
```bash
cd agentic-patterns
```

3. **Create and configure .env file**:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

4. **Install dependencies**:
```bash
uv sync
```

## Project Structure

```
agentic-patterns/
├── src/agentic_patterns/
│   ├── patterns/              # Individual pattern implementations
│   │   └── chaining_01/       # Chaining pattern example
│   ├── common/                # Shared utilities
│   │   ├── config.py          # Environment configuration
│   │   ├── model_factory.py   # LangChain model creation
│   │   └── output_writer.py   # Result logging
│   └── tools/                 # Custom tools
├── experiments/results/        # Log files (auto-created)
└── tests/                     # Test files
```

## Usage

### Running a Pattern

**Option 1: Direct execution**
```bash
# Run with default model (gpt-4)
uv run src/agentic_patterns/patterns/chaining_01/run.py

# Run with specific model
uv run src/agentic_patterns/patterns/chaining_01/run.py gpt-3.5-turbo
uv run src/agentic_patterns/patterns/chaining_01/run.py claude-sonnet-4
```

**Option 2: Import as module**
```bash
uv run python -c "from agentic_patterns.patterns.chaining_01 import run; run('gpt-4')"
```

### Comparing Multiple Models

Edit `run.py` and uncomment the comparison section, or use programmatically:

```python
from agentic_patterns.patterns.chaining_01 import compare_models

results = compare_models(
    models=["gpt-4", "gpt-3.5-turbo", "claude-sonnet-4"],
    input_text="Your custom input text here"
)
```

## Available Patterns

### 01_chaining
Sequential LLM calls using LangChain Expression Language (LCEL). Demonstrates:
- Prompt templates
- Chain composition
- Output parsing
- Sequential processing

## Output

All results are automatically written to timestamped log files in `experiments/results/`:
- Individual runs: `{pattern_name}_{timestamp}.log`
- Comparisons: `{pattern_name}_comparison_{timestamp}.log`

## Supported Models

- **OpenAI**: gpt-4, gpt-4-turbo, gpt-3.5-turbo
- **Anthropic**: claude-sonnet-4, claude-opus-4, claude-haiku-4
- **Google**: gemini-pro, gemini-1.5-pro

## Adding New Patterns

1. Create a new directory under `src/agentic_patterns/patterns/`
2. Add the following files:
   - `__init__.py` - Package initialization
   - `config.py` - Pattern-specific configuration
   - `run.py` - Main implementation
3. Use `ModelFactory` for model creation
4. Use `OutputWriter` for result logging

## Development

```bash
# Add a new dependency
uv add langchain-cohere

# Run tests (when available)
uv run pytest

# Format code
uv run ruff check src/
```

## License

MIT