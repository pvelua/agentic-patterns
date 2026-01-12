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

**Memory Management Pattern (memory_mgmt_08)**: Demonstrates short-term and long-term memory with multiple implementation approaches

- **Dependencies**: Requires `langchain-community` package for memory classes
  ```bash
  uv add langchain-community
  ```
- **Short-term Memory**:
  - `ConversationBufferMemory` from `langchain_classic.memory` (automated)
  - `ChatMessageHistory` from `langchain_community.chat_message_histories` (manual)
- **Long-term Memory Types**:
  - **Semantic Memory**: Facts and knowledge (e.g., user preferences, domain knowledge)
  - **Episodic Memory**: Past experiences and events (e.g., conversation history, past tickets)
  - **Procedural Memory**: Rules and strategies (e.g., policies, protocols)
- **Two Implementation Approaches**:
  1. **Simple (Examples 1-3)**: Lists/Dicts in config for demonstration
  2. **Production-style (Example 4)**: LangGraph's `InMemoryStore` with async operations
- **InMemoryStore Usage** (Example 4 - Finance Advisor):
  ```python
  from langgraph.store.memory import InMemoryStore

  # Create store with namespaces
  memory_store = InMemoryStore()

  # Store memories (async)
  await memory_store.aput(
      namespace=("advisor", "semantic"),
      key="tax_401k",
      value={"concept": "tax_401k", "knowledge": "..."}
  )

  # Retrieve specific memory (async)
  memory = await memory_store.aget(namespace=("advisor", "semantic"), key="tax_401k")

  # Search all memories in namespace (async)
  results = await memory_store.asearch(
      ("advisor", "episodic"),  # namespace_prefix (positional-only)
      limit=100
  )
  ```
- **Async Patterns**: Use `asyncio.run()` to execute async functions from main block
  ```python
  if example_name == "advisor":
      result = asyncio.run(run_finance_advisor(model_name=model_name, config=config))
  ```
- **Key Features**:
  - Namespace organization for different memory types
  - Async operations for scalable memory access
  - Document-based storage (JSON values)
  - Search capabilities with filters and limits
- **Usage Examples**:
  ```bash
  # Run all examples (uses simple Lists/Dicts approach)
  uv run src/agentic_patterns/patterns/memory_mgmt_08/run.py gpt-4o-mini all

  # Run specific example with InMemoryStore
  uv run src/agentic_patterns/patterns/memory_mgmt_08/run.py gpt-4o-mini advisor

  # Programmatic usage (async)
  uv run python -c "
  import asyncio
  from agentic_patterns.patterns.memory_mgmt_08 import run_finance_advisor
  result = asyncio.run(run_finance_advisor(model_name='gpt-4o-mini'))
  "
  ```
- **Benefits**: Shows progression from simple demo patterns to production-ready memory management

**Learn and Adapt Pattern (learn_adapt_09)**: Demonstrates agent self-improvement through iterative code modification

- **Use Case**: Data Processor Agent that improves its own implementation through iterative cycles
  - Starts with basic code that fails benchmark tests
  - Analyzes test failures and performance metrics
  - Uses LLM to generate improved versions
  - Selects best version based on weighted performance score
  - Continues until performance threshold reached or max iterations

- **Architecture Components**:
  ```python
  # Core classes for self-improvement
  class AgentVersion:
      """Stores code, test results, metrics, and score for each iteration"""
      version_id: int
      code: str
      test_results: Dict[str, Any]
      performance_metrics: Dict[str, float]
      overall_score: float

  class BenchmarkSuite:
      """Runs 4 benchmark tests and measures performance"""
      - test_numeric_processing: Compute statistics from numeric lists
      - test_edge_cases: Handle empty inputs, None values
      - test_string_operations: Word frequency counting
      - test_performance: Process large datasets efficiently

  class VersionArchive:
      """Manages version history and selects best performer"""
      - Stores all versions from all iterations
      - Calculates scores using weighted formula
      - Provides version history summary for LLM context

  class AdaptationEngine:
      """Uses LLM to analyze and improve code"""
      - Formats test results and failures for LLM
      - Provides version history as context
      - Extracts Python code from LLM response
  ```

- **Performance Scoring**: Weighted formula balancing three factors
  ```python
  score = (
      0.5 * success_rate +           # Test success rate (0-1)
      0.3 * (1 - normalized_time) +  # Execution speed (lower is better)
      0.2 * (1 - normalized_complexity)  # Code complexity (lines, lower is better)
  )
  ```

- **Iterative Improvement Cycle**:
  1. Execute current code on benchmark tests
  2. Calculate performance metrics (success rate, execution time, complexity)
  3. Calculate weighted overall score
  4. Store version in archive
  5. If threshold reached, stop; otherwise continue
  6. Generate improved version using LLM with:
     - Current code and test results
     - Failed test details
     - Version history summary
  7. Repeat from step 1 with improved code

- **Code Execution Pattern**: Safe code execution with isolated namespace
  ```python
  # Execute agent code in isolated namespace
  namespace = {}
  exec(code, namespace)
  process_data = namespace['process_data']

  # Run tests and measure execution time
  start_time = time.time()
  result = process_data(test_input)
  execution_time = time.time() - start_time
  ```

- **LLM Prompting Strategy**:
  - System prompt establishes LLM as code improvement expert
  - Provides current code, test results, and failure details
  - Includes version history to avoid repeating mistakes
  - Requests ONLY code output (no explanations)
  - Extracts code from markdown code blocks using regex

- **Usage Examples**:
  ```bash
  # Run with default model (gpt-4o) and 5 iterations
  uv run src/agentic_patterns/patterns/learn_adapt_09/run.py

  # Run with specific model
  uv run src/agentic_patterns/patterns/learn_adapt_09/run.py gpt-4o-mini

  # Compare multiple models
  uv run src/agentic_patterns/patterns/learn_adapt_09/run.py compare

  # Programmatic usage
  from agentic_patterns.patterns.learn_adapt_09 import run
  result = run(model_name='gpt-4o-mini', max_iterations=5)
  ```

- **Expected Results**: Progressive improvement trajectory
  - Iteration 0: ~0% success rate (basic implementation)
  - Iteration 1-2: ~67-83% success rate (fixes main issues)
  - Iteration 3-4: ~100% success rate (handles all edge cases)
  - Best version selected based on score (may not be final iteration)

- **Configuration Options**:
  ```python
  config = LearnAdaptConfig()
  config.max_iterations = 5              # Limit iteration count
  config.performance_threshold = 0.95    # Stop when 95% success rate
  config.weight_success = 0.5            # Success rate importance
  config.weight_time = 0.3               # Speed importance
  config.weight_complexity = 0.2         # Simplicity importance
  config.temperature = 0.3               # LLM creativity for code gen
  ```

- **Key Implementation Details**:
  - Uses `exec()` to dynamically execute generated code
  - Measures execution time with `time.time()` before/after
  - Calculates complexity as non-empty, non-comment lines
  - Handles code extraction from LLM responses (with/without markdown)
  - Provides detailed failure information to guide improvements
  - Tracks all versions to show improvement trajectory

- **Benefits**:
  - Demonstrates meta-learning (learning to improve oneself)
  - Shows how to safely execute and test dynamic code
  - Illustrates scoring multiple objectives (accuracy, speed, complexity)
  - Provides framework for agent self-optimization
  - Real-world applicable to ML model tuning, prompt engineering, hyperparameter search

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
- `langchain_core.prompts.ChatPromptTemplate`, `MessagesPlaceholder`
- `langchain_core.output_parsers.StrOutputParser`
- `langchain_core.messages.SystemMessage`, `HumanMessage`, `AIMessage`
- Model-specific: `langchain_openai.ChatOpenAI`, `langchain_anthropic.ChatAnthropic`, `langchain_google_genai.ChatGoogleGenerativeAI`
- Memory (requires langchain-community):
  - `langchain_classic.memory.ConversationBufferMemory`
  - `langchain_community.chat_message_histories.ChatMessageHistory`
- LangGraph: `langgraph.graph.StateGraph`, `langgraph.store.memory.InMemoryStore`

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
10. For patterns using async operations (e.g., InMemoryStore):
    - Mark functions as `async def` and use `await` for async calls
    - Use `asyncio.run()` in main block to execute async functions
    - Import `asyncio` at the top of the file
    - Update compare_models() to handle async functions with `asyncio.run()`
    - Example:
      ```python
      async def run_example(model_name: str = "gpt-4o"):
          memory_store = InMemoryStore()
          await memory_store.aput(namespace=("ns",), key="k", value={"data": "..."})
          # ... rest of async code

      if __name__ == "__main__":
          result = asyncio.run(run_example(model_name=model_name))
      ```
