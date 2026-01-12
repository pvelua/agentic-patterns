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

## Pattern 4 - Reflection

Running the Pattern:
- Basic usage with defaults:
```bash
uv run src/agentic_patterns/patterns/reflection_04/run.py gpt-4o
```
- Original single model implementation - Custom task with specific model:c
```bash
uv run python -c "from agentic_patterns.patterns.reflection_04 import run; result = run('gpt-4.1-2025-04-14', 'Your task here')"
```
- Two models implementation - a separate modle in the critic role:
```bash
uv run python -c "from agentic_patterns.patterns.reflection_04 import run; run('gpt-4.1-2025-04-14', 'claude-sonnet-4-5-20250929', 'Your task here')"
```
- Compare different model pairs:
```python
  from agentic_patterns.patterns.reflection_04 import compare_models

  compare_models(model_pairs=[
      ("gpt-4", "claude-sonnet-4-5-20250929"),
      ("gpt-4o", "claude-sonnet-4-5-20250929"),
      ("claude-sonnet-4-5-20250929", "gpt-4o"),
  ])
```

## Pattern 5 - Tool Use (Function Calling)

Usage Examples
```bash
  # Run with default model (gpt-4o) and example queries
  uv run src/agentic_patterns/patterns/tool_use_05/run.py

  # Run with specific model
  uv run src/agentic_patterns/patterns/tool_use_05/run.py claude-sonnet-4-5-20250929

  # Programmatic usage
  uv run python -c "from agentic_patterns.patterns.tool_use_05 import run; run('gpt-4o', ['What is quantum computing?'])"
```
  The implementation demonstrates sophisticated function calling where the LLM not only selects the right tool but also formulates comprehensive answers by synthesizing the tool results with its own knowledge!

## Pattern 6 - Planning

Usage Examples
```bash
  # Run with default task (RESTful API guide)
  uv run src/agentic_patterns/patterns/planning_06/run.py

  # Run with specific model
  uv run src/agentic_patterns/patterns/planning_06/run.py claude-sonnet-4-5-20250929

  # Programmatic usage
  uv run python -c "
  from agentic_patterns.patterns.planning_06 import run
  task = 'Design a microservices architecture for an e-commerce platform'
  run('gpt-4o', task)
  "
```

## Pattern 8 - Memory. Management

Usage Examples:
```bash
  # Run all examples
  uv run src/agentic_patterns/patterns/memory_mgmt_08/run.py gpt-4o-mini all

  # Run specific example
  uv run src/agentic_patterns/patterns/memory_mgmt_08/run.py gpt-4o-mini assistant
  uv run src/agentic_patterns/patterns/memory_mgmt_08/run.py gpt-4o-mini support
  uv run src/agentic_patterns/patterns/memory_mgmt_08/run.py gpt-4o-mini tutor

  # Compare models
  uv run python -c "from agentic_patterns.patterns.memory_mgmt_08 import compare_models; compare_models(example='assistant')"

  # Run finance advisor example
  uv run src/agentic_patterns/patterns/memory_mgmt_08/run.py gpt-4o-mini advisor

  # Programmatic usage
  uv run python -c "
  import asyncio
  from agentic_patterns.patterns.memory_mgmt_08 import run_finance_advisor

  result = asyncio.run(run_finance_advisor(
      model_name='gpt-4o-mini',
      client_question='Should I max out my 401(k) or open a Roth IRA?'
  ))
  print(result)
  "

  # Compare models on advisor example
  uv run python -c "
  from agentic_patterns.patterns.memory_mgmt_08 import compare_models
  compare_models(example='advisor')
  "
```

---

## Pattern 9 - Learn and Adapt

Usage Examples:
```bash
  # Run with default model (5 iterations)
  uv run src/agentic_patterns/patterns/learn_adapt_09/run.py gpt-4o-mini

  # Compare multiple models
  uv run src/agentic_patterns/patterns/learn_adapt_09/run.py compare

  # Programmatic usage
  from agentic_patterns.patterns.learn_adapt_09 import run
  result = run(model_name='gpt-4o', max_iterations=5)
```

---

## Pattern 10 - Goal Setting and Monitoring

  Usage Examples
```bash
  # Run with defaults (api_client goal, gpt-4o + claude-sonnet-4-5-20250929)
  uv run src/agentic_patterns/patterns/set_goal_monitor_10/run.py

  # Run specific goal
  uv run src/agentic_patterns/patterns/set_goal_monitor_10/run.py data_validator

  # Specify both models
  uv run src/agentic_patterns/patterns/set_goal_monitor_10/run.py task_scheduler gpt-4o-mini gpt-4o-mini

  # Compare model pairs
  uv run src/agentic_patterns/patterns/set_goal_monitor_10/run.py compare api_client
```
 Summary: How Model Selection Works

  1. Default Configuration (config.py:12-17):
  default_developer_model: str = "gpt-4o"              # OpenAI
  default_manager_model: str = "claude-sonnet-4-5-20250929"  # Anthropic

  2. CLI Usage:
```bash
  # Uses DEFAULTS (gpt-4o + claude-sonnet-4-5-20250929)
  uv run .../run.py api_client

  # Specify BOTH models explicitly
  uv run .../run.py api_client gpt-4o-mini claude-sonnet-4-5-20250929

  # What I tested earlier (both same for speed)
  uv run .../run.py api_client gpt-4o-mini gpt-4o-mini
```
  3. Separate LLM Instances (run.py):
```python
  developer_llm = ModelFactory.create(developer_model, **config.get_developer_kwargs())
  manager_llm = ModelFactory.create(manager_model, **config.get_manager_kwargs())
```
  4. Different Temperatures:
  - Developer: 0.2 (more deterministic for code generation)
  - Manager: 0.4 (more analytical for reviews)
---
