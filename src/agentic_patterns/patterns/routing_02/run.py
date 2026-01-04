"""
Routing Pattern - Delegating requests to specialized handlers using LangChain.

This pattern demonstrates:
1. Using an LLM as a coordinator to classify requests
2. Routing requests to different handlers based on classification
3. Using RunnableBranch for conditional execution
4. Simulating sub-agents with handler functions
"""

import warnings
# Suppress Pydantic v1 deprecation warning for Python 3.14+
warnings.filterwarnings('ignore', message='.*Pydantic V1.*', category=UserWarning)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

from agentic_patterns.common import ModelFactory, create_writer
from agentic_patterns.patterns.routing_02.config import RoutingConfig


# --- Define Simulated Sub-Agent Handlers ---
def booking_handler(request: str) -> str:
    """Simulates the Booking Agent handling a request."""
    return f"Booking Handler processed request: '{request}'. Result: Simulated booking action completed."


def info_handler(request: str) -> str:
    """Simulates the Info Agent handling a request."""
    return f"Info Handler processed request: '{request}'. Result: Simulated information retrieval completed."


def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be delegated."""
    return f"Coordinator could not delegate request: '{request}'. Please clarify your request."


def run(
    model_name: str = "gpt-4",
    request: str = None,
    config: RoutingConfig = None,
    verbose: bool = True
):
    """
    Run the routing pattern with specified model.

    This creates a coordinator that:
    1. Analyzes incoming requests
    2. Classifies them into categories (booker, info, unclear)
    3. Routes them to the appropriate handler

    Args:
        model_name: Model to use (e.g., 'gpt-4', 'claude-sonnet-4-5-20250929')
        request: User request to route
        config: Pattern configuration (uses defaults if None)
        verbose: Whether to print intermediate results

    Returns:
        Handler response as a string
    """
    # Use provided config or defaults
    config = config or RoutingConfig()

    # Default request if none provided
    if request is None:
        request = "Book me a flight to London."

    # Initialize the Language Model
    llm = ModelFactory.create(model_name, **config.get_model_kwargs())

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Running Routing Pattern with {model_name}")
        print(f"{'=' * 80}\n")

    # --- Build the Coordinator Router Chain ---
    # This chain decides which handler to delegate to
    coordinator_router_prompt = ChatPromptTemplate.from_messages([
        ("system", config.router_prompt),
        ("user", "{request}")
    ])

    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

    # --- Define the Delegation Logic using RunnableBranch ---
    # Define branches for different handler types
    branches = {
        "booker": RunnablePassthrough.assign(
            output=lambda x: booking_handler(x['request']['request'])
        ),
        "info": RunnablePassthrough.assign(
            output=lambda x: info_handler(x['request']['request'])
        ),
        "unclear": RunnablePassthrough.assign(
            output=lambda x: unclear_handler(x['request']['request'])
        ),
    }

    # Create the RunnableBranch to route based on the router's decision
    delegation_branch = RunnableBranch(
        (lambda x: x['decision'].strip() == 'booker', branches["booker"]),
        (lambda x: x['decision'].strip() == 'info', branches["info"]),
        branches["unclear"]  # Default branch
    )

    # Combine the router chain and delegation branch into a single runnable
    coordinator_agent = (
        {
            "decision": coordinator_router_chain,
            "request": RunnablePassthrough()
        }
        | delegation_branch
        | (lambda x: x['output'])  # Extract the final output
    )

    # --- Run the Coordinator Agent ---
    if verbose:
        print(f"User Request: {request}\n")
        print("Routing request to appropriate handler...")

    # Execute the coordinator agent
    final_result = coordinator_agent.invoke({"request": request})

    if verbose:
        print(f"\n--- Handler Response ---")
        print(final_result)
        print(f"\n{'=' * 80}\n")

    # Write results to log file
    writer = create_writer("02_routing")
    log_path = writer.write_result(
        result=final_result,
        model_name=model_name,
        input_data={"request": request},
        metadata={
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }
    )

    if verbose:
        print(f"Results written to: {log_path}\n")

    return final_result


def compare_models(
    models: list[str] = None,
    request: str = None,
    config: RoutingConfig = None
):
    """
    Compare the routing pattern across multiple models.

    Args:
        models: List of model names to compare
        request: User request to route
        config: Pattern configuration (uses defaults if None)

    Returns:
        Dictionary mapping model names to their results
    """
    if models is None:
        models = ["gpt-4", "gpt-4o", "claude-sonnet-4-5-20250929"]

    config = config or RoutingConfig()

    if request is None:
        request = "Book me a flight to London."

    print(f"\n{'=' * 80}")
    print(f"Comparing Routing Pattern Across Models")
    print(f"{'=' * 80}\n")

    results = {}
    for model in models:
        print(f"\nTesting with {model}...")
        try:
            result = run(
                model_name=model,
                request=request,
                config=config,
                verbose=False
            )
            results[model] = result
            print(f"✓ {model} completed successfully")
        except Exception as e:
            print(f"✗ {model} failed: {e}")
            results[model] = f"Error: {e}"

    # Write comparison results
    writer = create_writer("02_routing")
    log_path = writer.write_comparison(
        results=results,
        input_data={"request": request}
    )

    print(f"\n{'=' * 80}")
    print(f"Comparison results written to: {log_path}")
    print(f"{'=' * 80}\n")

    return results


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4"

    # Example requests
    booking_request = "Book me a flight to London."
    info_request = "What is the capital of Italy?"
    unclear_request = "Tell me about quantum physics."

    # Run with different request types
    print("\n=== Running with booking request ===")
    run(model_name=model, request=booking_request)

    print("\n=== Running with info request ===")
    run(model_name=model, request=info_request)

    print("\n=== Running with unclear request ===")
    run(model_name=model, request=unclear_request)

    # Compare multiple models (uncomment to use)
    # print("\n=== Comparing models ===")
    # compare_models(models=["gpt-4o", "claude-sonnet-4-5-20250929"])
