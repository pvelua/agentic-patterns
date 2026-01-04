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
uv run src/agentic_patterns/patterns/chaining_01/run.py gpt-4o
uv run src/agentic_patterns/patterns/chaining_01/run.py claude-sonnet-4-5-20250929
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
    models=["gpt-4o", "gpt-4.1-2025-04-14", "claude-sonnet-4-5-20250929"],
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

### OpenAI
**Current (2025):**
- `gpt-4.1-2025-04-14` - Latest GPT-4.1 with enhanced coding and 1M token context
- `gpt-4.1-mini-2025-04-14` - Cost-effective GPT-4.1 variant
- `gpt-4o` - Multimodal (voice, image, text) capabilities
- `gpt-4o-mini` - Faster, cheaper GPT-4o variant
- `gpt-4-turbo` - Older turbo model (available but superseded)

**Deprecated:** `gpt-3.5-turbo` (being phased out)

### Anthropic
**Current (2025):**
- `claude-sonnet-4-5-20250929` - Claude Sonnet 4.5
- `claude-opus-4-5-20251101` - Claude Opus 4.5 (most capable)
- `claude-3-5-haiku-20241022` - Claude 3.5 Haiku (fastest)

### Google Gemini
**Current (2025):**
- `gemini-3-pro-preview` - Latest Gemini 3 Pro (Nov 2025, no free tier)
- `gemini-3-flash-preview` - Latest Gemini 3 Flash (Nov 2025, free tier available)
- `gemini-2.5-pro` - Stable 2.5 Pro with adaptive thinking
- `gemini-2.5-flash` - Stable 2.5 Flash (best price-performance)
- `gemini-2.5-flash-lite` - Fast, low-cost variant

**Deprecated:** `gemini-pro`, `gemini-1.5-pro` (retired April 29, 2025)

### Sources
- OpenAI: [Models Documentation](https://platform.openai.com/docs/models), [GPT-4.1 Guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)
- Anthropic: [Models Overview](https://platform.claude.com/docs/en/about-claude/models/overview), [Migrating to Claude 4](https://docs.anthropic.com/en/docs/about-claude/models/migrating-to-claude-4)
- Google: [Gemini Models](https://ai.google.dev/gemini-api/docs/models), [Gemini 3 Guide](https://ai.google.dev/gemini-api/docs/gemini-3), [Release Notes](https://ai.google.dev/gemini-api/docs/changelog)

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