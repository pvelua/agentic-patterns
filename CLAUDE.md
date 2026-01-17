# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Experimental framework for testing various agentic design patterns using LangChain and LangGraph. The project implements 10 LLM agent patterns and allows comparison across different models (OpenAI GPT, Anthropic Claude, Google Gemini).

**Implemented Patterns:**
1. **Chaining (01)**: Sequential LLM calls with LCEL pipeline composition
2. **Routing (02)**: Request classification and conditional delegation using RunnableBranch
3. **Parallelization (03)**: Concurrent execution with RunnableParallel and result synthesis
4. **Reflection (04)**: Dual-model creator/critic iterative improvement cycle
5. **Tool Use (05)**: Function calling with domain-specific tools
6. **Planning (06)**: Two-phase plan generation and step-by-step execution using LangGraph
7. **Multi-Agent Collaboration (07)**: Three collaboration models (sequential, parallel+synthesis, debate) with LangGraph StateGraph
8. **Memory Management (08)**: Short/long-term memory with InMemoryStore (semantic, episodic, procedural)
9. **Learning & Adapting (09)**: Agent self-improvement through benchmark testing and iterative code generation
10. **Goal Setting & Monitoring (10)**: Developer/Manager agents with graded reviews and incremental improvement

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

**Routing Pattern (routing_02)**: Request classification and conditional execution with specialized handlers

- **Use Case**: Coordinator agent that analyzes requests and delegates to specialized handlers
  - Classifies requests into categories (booking, info, unclear)
  - Routes to appropriate handler based on classification
  - Uses RunnableBranch for conditional execution paths

- **Architecture Components**:
  ```python
  # Handler functions for different request types
  def booking_handler(request: str) -> str:
      """Processes booking requests"""

  def info_handler(request: str) -> str:
      """Handles information retrieval"""

  def unclear_handler(request: str) -> str:
      """Handles unclear/ambiguous requests"""
  ```

- **Key LangChain Constructs**:
  - `RunnableBranch`: Conditional routing based on classification
  - Classification prompt extracts category from user request
  - Routing logic: "booker" → booking_handler, "info" → info_handler, default → unclear_handler

- **Usage Examples**:
  ```bash
  # Run with default model and request
  uv run src/agentic_patterns/patterns/routing_02/run.py

  # Run with specific model
  uv run src/agentic_patterns/patterns/routing_02/run.py gpt-4o

  # Programmatic usage
  from agentic_patterns.patterns.routing_02 import run
  result = run(model_name='gpt-4', request='Book a flight to Paris')
  ```

- **Benefits**: Demonstrates conditional execution, request classification, and modular handler architecture

**Parallelization Pattern (parallelization_03)**: Concurrent execution with result synthesis

- **Use Case**: Execute multiple independent analysis tasks in parallel and synthesize results
  - Runs 3 parallel chains: summary, questions, key terms
  - Reduces total execution time through concurrency
  - Synthesizes all results into comprehensive output

- **Architecture Components**:
  ```python
  # Three parallel chains
  parallel_chains = RunnableParallel(
      summary=summary_chain,       # Summarize topic
      questions=questions_chain,   # Generate questions
      key_terms=key_terms_chain    # Extract key terms
  )

  # Synthesis chain combines parallel results
  full_chain = parallel_chains | synthesis_chain
  ```

- **Async Execution**: Pattern uses async/await for true parallelization
  ```python
  async def run(model_name: str = "gpt-4", topic: str = None):
      result = await full_chain.ainvoke({"topic": topic})
      return result

  # Main block
  if __name__ == "__main__":
      result = asyncio.run(run(model_name=model_name, topic=topic))
  ```

- **Usage Examples**:
  ```bash
  # Run with default model and topic
  uv run src/agentic_patterns/patterns/parallelization_03/run.py

  # Run with specific model
  uv run src/agentic_patterns/patterns/parallelization_03/run.py claude-sonnet-4-5-20250929

  # Programmatic usage (async)
  import asyncio
  from agentic_patterns.patterns.parallelization_03 import run
  result = asyncio.run(run(model_name='gpt-4o', topic='Quantum Computing'))
  ```

- **Benefits**: Shows how to parallelize independent operations, reduce latency, and synthesize multiple perspectives

**Tool Use Pattern (tool_use_05)**: Function calling with domain-specific tools

- **Use Case**: Agent with access to specialized search tools for different domains
  - Defines 3 tools: tech_search, science_search, history_search
  - Agent selects appropriate tool based on query content
  - Executes tool and uses results to formulate answer

- **Tool Definition Pattern**:
  ```python
  from langchain_core.tools import tool as langchain_tool

  @langchain_tool
  def tech_search(query: str) -> str:
      """Search for technology-related information."""
      return f"Tech search results for: {query}"

  @langchain_tool
  def science_search(query: str) -> str:
      """Search for science-related information."""
      return f"Science search results for: {query}"
  ```

- **Tool Binding and Execution**:
  ```python
  # Bind tools to model
  tools = [tech_search, science_search, history_search]
  llm_with_tools = llm.bind_tools(tools)

  # Process tool calls from model response
  if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
      for tool_call in ai_msg.tool_calls:
          tool_result = execute_tool(tool_call)
          messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
  ```

- **Usage Examples**:
  ```bash
  # Run with default queries
  uv run src/agentic_patterns/patterns/tool_use_05/run.py

  # Run with specific model
  uv run src/agentic_patterns/patterns/tool_use_05/run.py gpt-4o

  # Programmatic usage
  from agentic_patterns.patterns.tool_use_05 import run
  result = run(
      model_name='claude-sonnet-4-5-20250929',
      queries=['What is machine learning?', 'Who built the pyramids?']
  )
  ```

- **Benefits**: Demonstrates function calling, tool selection logic, and multi-turn tool-using conversations

**Planning Pattern (planning_06)**: Two-phase strategic planning with execution using LangGraph

- **Use Case**: Break down complex tasks into detailed plans, then execute step-by-step
  - Phase 1: Planner agent analyzes task and creates structured plan
  - Phase 2: Executor agent follows plan steps sequentially
  - Uses LangGraph StateGraph for workflow management

- **Architecture Components**:
  ```python
  from langgraph.graph import StateGraph, END

  class PlanningState(TypedDict):
      task: str          # Original task
      plan: str          # Generated plan
      execution: str     # Execution result
      messages: list     # Message history

  # Build workflow graph
  workflow = StateGraph(PlanningState)
  workflow.add_node("planner", planner_agent)    # Creates plan
  workflow.add_node("executor", executor_agent)  # Executes plan
  workflow.add_edge("planner", "executor")       # Sequential flow
  workflow.add_edge("executor", END)
  ```

- **Two-Agent System**:
  - **Planner Agent**: Analyzes task, breaks into steps, creates detailed plan
  - **Executor Agent**: Receives plan, executes each step, produces final result

- **Usage Examples**:
  ```bash
  # Run with default task
  uv run src/agentic_patterns/patterns/planning_06/run.py

  # Run with specific model
  uv run src/agentic_patterns/patterns/planning_06/run.py gpt-4o

  # Programmatic usage
  from agentic_patterns.patterns.planning_06 import run
  result = run(
      model_name='claude-sonnet-4-5-20250929',
      task='Design a database schema for an e-commerce platform'
  )
  ```

- **Benefits**: Shows strategic planning, step-by-step execution, and LangGraph workflow orchestration

**Multi-Agent Collaboration Pattern (multi_agent_collab_07)**: Three collaboration models using LangGraph

- **Use Case**: Demonstrates different ways multiple agents can work together
  1. **Sequential Pipeline**: Research → Critic → Summarizer (paper analysis)
  2. **Parallel + Synthesis**: Marketing + Content + Analyst → Coordinator (product launch)
  3. **Multi-Perspective Debate**: Security + Performance + Quality → Synthesizer (code review)

- **Example 1: Sequential Pipeline (Research Paper Analysis)**:
  ```python
  class ResearchAnalysisState(TypedDict):
      paper_content: str
      research_analysis: str     # Researcher output
      critical_review: str       # Critic output
      final_summary: str         # Summarizer output
      messages: list

  workflow = StateGraph(ResearchAnalysisState)
  workflow.add_node("researcher", researcher_agent)
  workflow.add_node("critic", critic_agent)
  workflow.add_node("summarizer", summarizer_agent)
  workflow.add_edge("researcher", "critic")
  workflow.add_edge("critic", "summarizer")
  workflow.add_edge("summarizer", END)
  ```

- **Example 2: Parallel + Synthesis (Product Launch Campaign)**:
  ```python
  # Three agents work in parallel
  workflow.add_node("marketing", marketing_agent)
  workflow.add_node("content", content_agent)
  workflow.add_node("analyst", analyst_agent)

  # All converge to coordinator
  workflow.add_edge("marketing", "coordinator")
  workflow.add_edge("content", "coordinator")
  workflow.add_edge("analyst", "coordinator")
  ```

- **Example 3: Multi-Perspective Debate (Code Review)**:
  ```python
  # Three reviewers provide different perspectives
  workflow.add_node("security_reviewer", security_agent)
  workflow.add_node("performance_reviewer", performance_agent)
  workflow.add_node("quality_reviewer", quality_agent)
  workflow.add_node("synthesizer", synthesizer_agent)

  # All reviews feed to synthesizer
  workflow.add_edge("security_reviewer", "synthesizer")
  workflow.add_edge("performance_reviewer", "synthesizer")
  workflow.add_edge("quality_reviewer", "synthesizer")
  ```

- **Usage Examples**:
  ```bash
  # Run sequential pipeline (research paper analysis)
  uv run src/agentic_patterns/patterns/multi_agent_collab_07/run.py research

  # Run parallel + synthesis (product launch)
  uv run src/agentic_patterns/patterns/multi_agent_collab_07/run.py campaign

  # Run debate (code review)
  uv run src/agentic_patterns/patterns/multi_agent_collab_07/run.py code_review

  # Run all examples
  uv run src/agentic_patterns/patterns/multi_agent_collab_07/run.py all

  # Specify model
  uv run src/agentic_patterns/patterns/multi_agent_collab_07/run.py research gpt-4o
  ```

- **Benefits**: Demonstrates multiple collaboration patterns, LangGraph StateGraph orchestration, and agent specialization strategies

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

**Goal Setting and Monitoring Pattern (set_goal_monitor_10)**: Two-agent collaborative development with iterative reviews

- **Architecture**: Two specialized agents with different LLM models working together
  - **Developer Agent**: Creates implementation plans and generates Python code
  - **Manager Agent**: Monitors progress, reviews code quality, and provides feedback
  - **Iterative Cycle**: Developer improves based on manager's detailed reviews

- **Agent Roles and Responsibilities**:
  ```python
  # Developer Agent
  - Analyzes user requirements
  - Creates detailed implementation plan
  - Writes production-ready Python code
  - Receives manager feedback
  - Improves code iteratively

  # Manager Agent
  - Reviews code against requirements
  - Evaluates code quality and best practices
  - Assesses error handling and edge cases
  - Grades documentation quality
  - Provides actionable feedback
  ```

- **Grading System**: Weighted scoring across four criteria (0-100 scale)
  ```python
  Grade = (
      40% * Requirements Coverage +  # All requirements met
      30% * Code Quality +            # Structure, readability, PEP 8
      15% * Error Handling +          # Robustness, edge cases
      15% * Documentation             # Docstrings, clarity
  )
  ```

- **Sample User Goals** (3 pre-configured scenarios):
  1. **REST API Client**: HTTP client with retry logic, rate limiting, error handling
  2. **Data Validator**: Validation library with custom rules and nested object support
  3. **Task Scheduler**: Priority-based scheduling with cron-like syntax and dependencies

- **Iterative Improvement Cycle**:
  1. Developer creates implementation plan (iteration 0)
  2. Developer generates code based on plan
  3. Manager reviews code and assigns grade
  4. If grade ≥ passing threshold (85/100), stop
  5. Manager provides detailed feedback on issues
  6. Developer improves code addressing all feedback
  7. Repeat steps 3-6 up to max iterations (default 4)

- **Review Format**: Structured feedback for systematic improvement
  ```
  GRADE: <score>/100

  REQUIREMENTS COVERAGE (<score>/40):
  - Requirement 1: Met/Partially Met/Not Met - explanation
  - Requirement 2: Met/Partially Met/Not Met - explanation
  ...

  CODE QUALITY (<score>/30):
  <Assessment of structure, readability, best practices>

  ERROR HANDLING (<score>/15):
  <Assessment of robustness and edge cases>

  DOCUMENTATION (<score>/15):
  <Assessment of docstrings and clarity>

  OVERALL ASSESSMENT:
  <Summary of strengths and weaknesses>

  FEEDBACK FOR IMPROVEMENT:
  1. <Specific actionable item>
  2. <Specific actionable item>
  ...
  ```

- **Configuration Options**:
  ```python
  config = GoalMonitorConfig()
  config.default_developer_model = "gpt-4o"
  config.default_manager_model = "claude-sonnet-4-5-20250929"
  config.developer_temperature = 0.2   # Deterministic code gen
  config.manager_temperature = 0.4     # Analytical review
  config.max_iterations = 7            # Increased from 4 (Jan 2026)
  config.passing_grade = 85.0          # Stop when reached
  ```

- **Usage Examples**:
  ```bash
  # Run with default models and api_client goal
  uv run src/agentic_patterns/patterns/set_goal_monitor_10/run.py

  # Run specific goal with default models
  uv run src/agentic_patterns/patterns/set_goal_monitor_10/run.py data_validator

  # Specify both developer and manager models
  uv run src/agentic_patterns/patterns/set_goal_monitor_10/run.py api_client gpt-4o claude-sonnet-4-5-20250929

  # Compare different model pairs
  uv run src/agentic_patterns/patterns/set_goal_monitor_10/run.py compare api_client

  # Programmatic usage
  from agentic_patterns.patterns.set_goal_monitor_10 import run
  result = run(
      goal_name="task_scheduler",
      developer_model="gpt-4o",
      manager_model="claude-sonnet-4-5-20250929",
      max_iterations=7  # Default is 7 (Jan 2026 update)
  )
  ```

- **Expected Grade Progression**:
  - Iteration 1: 70-85/100 (initial implementation, some missing features)
  - Iteration 2: 80-90/100 (addresses major feedback, adds missing features)
  - Iteration 3-4: 85-95/100 (polishes documentation, improves error handling)

- **Enhanced Version (January 2026)**: Monotonic improvement features
  - **Problem Solved**: Original version showed grade plateaus and occasional drops (e.g., 62→72→72→78 or 78→78→72)
  - **Solution**: "Carry-Forward Best Code" + "Incremental Requirements" approach

  **Key Enhancements**:
  1. **Best Code Tracking** (run.py:313-401)
     - Tracks best code version across all iterations
     - Always provides best code as starting point for next iteration
     - Prevents starting from failed attempts
     - Shows clear feedback: `✓ Grade improved: 78.0 → 82.0` or `⚠ Grade dropped from best: 62.0`

  2. **Prioritized Feedback** (config.py:264-271)
     - Manager separates feedback into:
       - **PRIORITY FEEDBACK**: Critical blocking issues (missing requirements, bugs)
       - **SECONDARY IMPROVEMENTS**: Nice-to-have enhancements (docs, refactoring)
     - Ensures developer addresses critical issues first

  3. **Incremental Improvement Instructions** (config.py:177-183)
     - Developer explicitly told to:
       - Start from provided code (not rewrite from scratch)
       - Preserve working functionality
       - Make incremental changes
       - Address priority items first before secondary improvements

  4. **Increased Iteration Limit**
     - Max iterations: 4 → 7 (allows more exploration and refinement)

  5. **Enhanced Progress Tracking**
     - Shows grade progression: `62.0 → 72.0 → 78.0 → 78.0 → 78.0 → 78.0 → 82.0`
     - Displays total improvement: `+20.0 points`
     - Identifies best iteration even if not final iteration

  **Test Results** (January 2026):
  | Goal | Initial | Final | Improvement | Progression |
  |------|---------|-------|-------------|-------------|
  | data_validator | 72 | 82 | +10 pts | 72→68→62→62→78→82→82 |
  | api_client | 62 | 82 | +20 pts | 62→72→78→78→78→78→82 |
  | task_scheduler | 42 | 62 | +20 pts | 42→52→52→58→58→62→62 |

  **Benefits**:
  - Prevents cascading failures when improvements backfire
  - Enables exploration (developer can try risky approaches) with safety net
  - Ensures monotonic or stable improvement (grades never significantly regress)
  - Provides clearer feedback on what's working vs. what's not

- **Key Implementation Details**:
  - Developer uses planning prompt first, then implementation prompt
  - Developer receives best code version as starting point for improvements (Jan 2026)
  - Manager uses structured review prompt with explicit grading rubric
  - Manager provides prioritized feedback (priority vs. secondary) (Jan 2026)
  - Feedback extraction via regex from manager's review
  - Code cleaning to handle markdown-wrapped responses
  - Iteration tracking with full history (plan, code, review, grade)
  - Best code tracking across iterations prevents regression (Jan 2026)
  - Best iteration selection based on highest grade

- **Benefits**:
  - Demonstrates multi-agent collaboration patterns
  - Shows how different models can specialize (creator vs. reviewer)
  - Illustrates structured feedback loops for improvement
  - Provides framework for quality assurance workflows
  - Real-world applicable to code review automation, automated testing, development workflows

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
- **Prompts & Messages**:
  - `langchain_core.prompts.ChatPromptTemplate`, `MessagesPlaceholder`
  - `langchain_core.messages.SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`
- **Output Parsers**: `langchain_core.output_parsers.StrOutputParser`
- **Runnables** (LCEL):
  - `langchain_core.runnables.RunnablePassthrough` - Pass data through chains
  - `langchain_core.runnables.RunnableBranch` - Conditional routing (routing_02)
  - `langchain_core.runnables.RunnableParallel` - Parallel execution (parallelization_03)
- **Tools**: `langchain_core.tools.tool` decorator for function definitions (tool_use_05)
- **Model Classes**: `langchain_openai.ChatOpenAI`, `langchain_anthropic.ChatAnthropic`, `langchain_google_genai.ChatGoogleGenerativeAI`
- **Memory** (requires langchain-community):
  - `langchain_classic.memory.ConversationBufferMemory`
  - `langchain_community.chat_message_histories.ChatMessageHistory`
- **LangGraph** (workflow orchestration):
  - `langgraph.graph.StateGraph` - State machine for agent workflows (planning_06, multi_agent_collab_07)
  - `langgraph.graph.END` - Terminal node marker
  - `langgraph.store.memory.InMemoryStore` - Production memory storage (memory_mgmt_08)

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
