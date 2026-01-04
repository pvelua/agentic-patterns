# Use Guide - Running Workflow Patterns

## Pattern 3 - Parallelization

Here's how to run parallelization_03 with a custom topic and the gpt-4.1-2025-04-14 model:
```bash
  uv run python -c "import asyncio; from agentic_patterns.patterns.parallelization_03 import run; asyncio.run(run('gpt-4.1-2025-04-14', 'The evolution of RAG technology'))"
```
  Explanation:
  - 'gpt-4.1-2025-04-14' - the model name (you selected from the README)
  - 'The evolution of RAG technology' - custom topic (replace with any topic you want)

  More examples:
```bash
  # Example with Claude Sonnet 4.5
  uv run python -c "import asyncio; from agentic_patterns.patterns.parallelization_03 import run; asyncio.run(run('claude-sonnet-4-5-20250929', 'The future of biotechnology'))"

  # Example with Gemini
  uv run python -c "import asyncio; from agentic_patterns.patterns.parallelization_03 import run; asyncio.run(run('gemini-2.5-flash', 'Ocean conservation and marine biodiversity'))"

  # Example with GPT-4o
  uv run python -c "import asyncio; from agentic_patterns.patterns.parallelization_03 import run; asyncio.run(run('gpt-4o', 'The impact of social media on society'))"
```
  Note: The topic should be enclosed in quotes. If your topic contains quotes, use different quote styles:
```bash
  uv run python -c "import asyncio; from agentic_patterns.patterns.parallelization_03 import run; asyncio.run(run('gpt-4.1-2025-04-14', \"The role of AI in healthcare\"))"
```
  The pattern will run the three parallel chains (summarize, questions, key terms) on your custom topic and synthesize the results.