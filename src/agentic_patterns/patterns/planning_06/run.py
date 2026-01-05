"""
Planning Pattern - Multi-step plan generation and execution using LangGraph.

This pattern demonstrates:
1. Strategic planning: Breaking down complex tasks into steps
2. Sequential execution: Following the plan step by step
3. State management: Tracking progress through LangGraph
4. Two-phase approach: Plan first, then execute
"""

import warnings
# Suppress Pydantic v1 deprecation warning for Python 3.14+
warnings.filterwarnings('ignore', message='.*Pydantic V1.*', category=UserWarning)

from typing import TypedDict, Annotated
import operator
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from agentic_patterns.common import ModelFactory, create_writer
from agentic_patterns.patterns.planning_06.config import PlanningConfig


# Define the state structure for the planning workflow
class PlanningState(TypedDict):
    """State for the planning workflow."""
    task: str  # The original task
    plan: str  # The generated plan
    execution: str  # The execution result
    messages: Annotated[list, operator.add]  # Message history


def run(
    model_name: str = "gpt-4o",
    task: str = None,
    config: PlanningConfig = None,
    verbose: bool = True
):
    """
    Run the planning pattern with a specified model and task.

    This creates a workflow that:
    1. Analyzes the task and creates a detailed plan
    2. Executes the plan step by step
    3. Produces a comprehensive result

    Args:
        model_name: Model to use for both planning and execution (default: gpt-4o)
        task: Task description to plan and execute (uses default if None)
        config: Pattern configuration (uses defaults if None)
        verbose: Whether to print intermediate results

    Returns:
        Dictionary containing the plan, execution result, and metadata
    """
    # Use provided config or defaults
    config = config or PlanningConfig()

    # Default task if none provided
    if task is None:
        task = """Create a comprehensive guide on implementing a RESTful API for a book library system.
The guide should cover design principles, best practices, and include example endpoints."""

    # Initialize the Language Models
    planner_llm = ModelFactory.create(model_name, **config.get_planner_kwargs())
    executor_llm = ModelFactory.create(model_name, **config.get_executor_kwargs())

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Running Planning Pattern")
        print(f"Model: {model_name}")
        print(f"Task: {task[:100]}..." if len(task) > 100 else f"Task: {task}")
        print(f"{'=' * 80}\n")

    # --- Define Workflow Nodes ---

    def planning_node(state: PlanningState) -> PlanningState:
        """Generate a detailed plan for the task."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("PHASE 1: PLANNING")
            print(f"{'─' * 80}")

        # Create planning prompt
        planning_messages = [
            SystemMessage(content=config.planner_system_prompt),
            HumanMessage(content=f"Task: {state['task']}\n\nPlease create a detailed plan for this task.")
        ]

        # Generate plan
        response = planner_llm.invoke(planning_messages)
        plan = response.content

        if verbose:
            print("\n📋 Generated Plan:")
            print(plan)

        return {
            **state,
            "plan": plan,
            "messages": planning_messages + [response]
        }

    def execution_node(state: PlanningState) -> PlanningState:
        """Execute the plan step by step."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("PHASE 2: EXECUTION")
            print(f"{'─' * 80}")

        # Create execution prompt
        execution_messages = [
            SystemMessage(content=config.executor_system_prompt),
            HumanMessage(content=f"""Original Task: {state['task']}

Plan to Execute:
{state['plan']}

Please execute this plan thoroughly, following each step in order and providing detailed results.""")
        ]

        # Execute plan
        response = executor_llm.invoke(execution_messages)
        execution = response.content

        if verbose:
            print("\n✅ Execution Result:")
            print(execution)

        return {
            **state,
            "execution": execution,
            "messages": state["messages"] + execution_messages + [response]
        }

    # --- Build the Workflow Graph ---
    workflow = StateGraph(PlanningState)

    # Add nodes
    workflow.add_node("planner", planning_node)
    workflow.add_node("executor", execution_node)

    # Define edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", END)

    # Compile the graph
    app = workflow.compile()

    # --- Execute the Workflow ---
    if verbose:
        print(f"\n{'═' * 80}")
        print("STARTING WORKFLOW")
        print(f"{'═' * 80}")

    # Initialize state
    initial_state = {
        "task": task,
        "plan": "",
        "execution": "",
        "messages": []
    }

    # Run the workflow
    final_state = app.invoke(initial_state)

    # --- Extract Results ---
    result = {
        "task": task,
        "plan": final_state["plan"],
        "execution": final_state["execution"],
        "workflow_steps": 2  # planning + execution
    }

    if verbose:
        print(f"\n{'═' * 80}")
        print("WORKFLOW COMPLETE")
        print(f"{'═' * 80}\n")

    # Write results to log file
    writer = create_writer("06_planning")
    log_path = writer.write_result(
        result=result,
        model_name=model_name,
        input_data={"task": task},
        metadata={
            "planner_temperature": config.planner_temperature,
            "planner_max_tokens": config.planner_max_tokens,
            "executor_temperature": config.executor_temperature,
            "executor_max_tokens": config.executor_max_tokens,
            "workflow_steps": result["workflow_steps"]
        }
    )

    if verbose:
        print(f"Results written to: {log_path}\n")

    return result


def compare_models(
    model_names: list[str] = None,
    task: str = None,
    config: PlanningConfig = None
):
    """
    Compare the planning pattern across multiple models.

    Args:
        model_names: List of model names to compare
        task: Task description (uses default if None)
        config: Pattern configuration (uses defaults if None)

    Returns:
        Dictionary mapping model names to their results
    """
    if model_names is None:
        model_names = [
            "gpt-4o",
            "claude-sonnet-4-5-20250929",
            "gpt-4o-mini",
        ]

    config = config or PlanningConfig()

    if task is None:
        task = """Design a data pipeline for processing and analyzing customer feedback from multiple sources
(social media, surveys, support tickets). Include data collection, processing, storage, and visualization."""

    print(f"\n{'=' * 80}")
    print(f"Comparing Planning Pattern Across Models")
    print(f"{'=' * 80}\n")

    results = {}
    for model in model_names:
        print(f"\nTesting with {model}...")
        try:
            result = run(
                model_name=model,
                task=task,
                config=config,
                verbose=False
            )
            results[model] = result
            plan_lines = result["plan"].count('\n')
            exec_lines = result["execution"].count('\n')
            print(f"✓ Completed successfully (Plan: {plan_lines} lines, Execution: {exec_lines} lines)")
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[model] = f"Error: {e}"

    # Write comparison results
    writer = create_writer("06_planning")
    log_path = writer.write_comparison(
        results=results,
        input_data={"task": task}
    )

    print(f"\n{'=' * 80}")
    print(f"Comparison results written to: {log_path}")
    print(f"{'=' * 80}\n")

    return results


if __name__ == "__main__":
    import sys

    # Parse command line argument for model name
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o"

    # Example tasks demonstrating different planning scenarios

    # Task 1: Software Development
    task1 = """Create a comprehensive guide on implementing a RESTful API for a book library system.
The guide should cover:
- API design principles and best practices
- Authentication and authorization
- CRUD operations for books, authors, and users
- Search and filtering capabilities
- Rate limiting and caching strategies
- Error handling and validation
- Example endpoints with request/response formats
- Testing strategies"""

    # Task 2: Data Analysis
    task2 = """Design and implement a data pipeline for analyzing e-commerce sales data.
Requirements:
- Data collection from multiple sources (database, API, CSV files)
- Data cleaning and transformation
- Feature engineering for customer segmentation
- Statistical analysis and trend identification
- Machine learning model for sales prediction
- Interactive dashboard for visualization
- Automated reporting system
- Scalability considerations"""

    # Task 3: Technical Writing
    task3 = """Write a tutorial on building a real-time chat application using WebSockets.
The tutorial should include:
- Architecture overview and design decisions
- Backend implementation (Node.js/Python)
- Frontend implementation (React/Vue)
- WebSocket connection management
- Message broadcasting and room management
- User authentication and presence tracking
- Error handling and reconnection logic
- Deployment and scaling considerations
- Code examples and best practices"""

    # Task 4: Problem Solving
    task4 = """Solve the following optimization problem:
A delivery company needs to optimize its route planning for 20 delivery locations in a city.
Constraints:
- Each vehicle can carry maximum 15 packages
- Deliveries must be completed within an 8-hour window
- Some locations have time-specific delivery windows
- Traffic patterns vary throughout the day
- Fuel efficiency needs to be optimized

Provide:
- Problem formulation
- Algorithm selection and justification
- Implementation approach
- Performance optimization strategies
- Testing methodology"""

    print(f"\n{'=' * 80}")
    print(f"Planning Pattern - Example Executions")
    print(f"{'=' * 80}\n")

    # Run with task 1 (Software Development)
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Software Development Task")
    print("=" * 80)
    run(model_name=model, task=task1)

    # Uncomment to run more examples
    # print("\n" + "=" * 80)
    # print("EXAMPLE 2: Data Analysis Task")
    # print("=" * 80)
    # run(model_name=model, task=task2)

    # print("\n" + "=" * 80)
    # print("EXAMPLE 3: Technical Writing Task")
    # print("=" * 80)
    # run(model_name=model, task=task3)

    # Uncomment to run model comparison
    # print("\n\n=== Comparing Models ===")
    # compare_models()
