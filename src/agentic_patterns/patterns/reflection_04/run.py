"""
Reflection Pattern - Iterative self-improvement through critique and refinement.

This pattern demonstrates:
1. Agent evaluating its own work (self-correction)
2. Iterative refinement based on internal critique
3. Using separate roles: Producer (generator) and Critic (reflector)
4. Stopping condition based on quality assessment
"""

import warnings
# Suppress Pydantic v1 deprecation warning for Python 3.14+
warnings.filterwarnings('ignore', message='.*Pydantic V1.*', category=UserWarning)

from langchain_core.messages import SystemMessage, HumanMessage

from agentic_patterns.common import ModelFactory, create_writer
from agentic_patterns.patterns.reflection_04.config import ReflectionConfig


def run(
    creator_model: str = None,
    critic_model: str = None,
    task_prompt: str = None,
    config: ReflectionConfig = None,
    verbose: bool = True
):
    """
    Run the reflection pattern with specified models for creator and critic roles.

    This creates a reflection loop that:
    1. Generates initial code based on task prompt (using creator model)
    2. Critiques the generated code (using critic model)
    3. Refines the code based on critique (using creator model)
    4. Repeats until code is perfect or max iterations reached

    Args:
        creator_model: Model to use for code generation (default: gpt-4)
        critic_model: Model to use for code critique (default: claude-sonnet-4-5-20250929)
        task_prompt: Task description for code generation
        config: Pattern configuration (uses defaults if None)
        verbose: Whether to print intermediate results

    Returns:
        Dictionary containing final code and iteration details
    """
    # Use provided config or defaults
    config = config or ReflectionConfig()

    # Default task if none provided
    if task_prompt is None:
        task_prompt = config.task_prompt

    # Use default models if not provided
    if creator_model is None:
        creator_model = config.default_creator_model
    if critic_model is None:
        critic_model = config.default_critic_model

    # Initialize separate Language Models for creator and critic
    creator_llm = ModelFactory.create(creator_model, **config.get_creator_kwargs())
    critic_llm = ModelFactory.create(critic_model, **config.get_critic_kwargs())

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Running Reflection Pattern")
        print(f"Creator Model: {creator_model}")
        print(f"Critic Model: {critic_model}")
        print(f"{'=' * 80}\n")

    # --- The Reflection Loop ---
    current_code = ""
    iterations_log = []

    # Build conversation history to provide context in each step
    message_history = [HumanMessage(content=task_prompt)]

    for i in range(config.max_iterations):
        if verbose:
            print("\n" + "=" * 25 + f" REFLECTION LOOP: ITERATION {i + 1} " + "=" * 25)

        # --- STAGE 1: GENERATE / REFINE ---
        if i == 0:
            if verbose:
                print(f"\n>>> STAGE 1: GENERATING initial code (using {creator_model})...")
            # First iteration: generate initial code
            response = creator_llm.invoke(message_history)
            current_code = response.content
        else:
            if verbose:
                print(f"\n>>> STAGE 1: REFINING code based on previous critique (using {creator_model})...")
            # Subsequent iterations: refine based on critique
            message_history.append(
                HumanMessage(content="Please refine the code using the critiques provided.")
            )
            response = creator_llm.invoke(message_history)
            current_code = response.content

        if verbose:
            print(f"\n--- Generated Code (v{i + 1}) ---")
            print(current_code)

        # Add the generated code to history
        message_history.append(response)

        # --- STAGE 2: REFLECT ---
        if verbose:
            print(f"\n>>> STAGE 2: REFLECTING on the generated code (using {critic_model})...")

        # Create reflector prompt with critic role
        reflector_prompt = [
            SystemMessage(content=config.reflector_system_prompt),
            HumanMessage(content=f"Original Task:\n{task_prompt}\n\nCode to Review:\n{current_code}")
        ]

        critique_response = critic_llm.invoke(reflector_prompt)
        critique = critique_response.content

        # Log this iteration
        iterations_log.append({
            'iteration': i + 1,
            'code': current_code,
            'critique': critique
        })

        # --- STOPPING CONDITION ---
        if config.stopping_phrase in critique:
            if verbose:
                print("\n--- Critique ---")
                print("No further critiques found. The code is satisfactory.")
            break

        if verbose:
            print("\n--- Critique ---")
            print(critique)

        # Add critique to history for next refinement loop
        message_history.append(
            HumanMessage(content=f"Critique of the previous code:\n{critique}")
        )

    # --- Final Result ---
    if verbose:
        print("\n" + "=" * 30 + " FINAL RESULT " + "=" * 30)
        print("\nFinal refined code after the reflection process:\n")
        print(current_code)
        print(f"\n{'=' * 80}\n")

    result = {
        'final_code': current_code,
        'iterations': len(iterations_log),
        'iterations_log': iterations_log
    }

    # Write results to log file
    writer = create_writer("04_reflection")
    log_path = writer.write_result(
        result=result,
        model_name=f"{creator_model} (creator) + {critic_model} (critic)",
        input_data={"task_prompt": task_prompt},
        metadata={
            "creator_model": creator_model,
            "critic_model": critic_model,
            "creator_temperature": config.creator_temperature,
            "creator_max_tokens": config.creator_max_tokens,
            "critic_temperature": config.critic_temperature,
            "critic_max_tokens": config.critic_max_tokens,
            "max_iterations": config.max_iterations
        }
    )

    if verbose:
        print(f"Results written to: {log_path}\n")

    return result


def compare_models(
    model_pairs: list[tuple[str, str]] = None,
    task_prompt: str = None,
    config: ReflectionConfig = None
):
    """
    Compare the reflection pattern across multiple model pairs.

    Args:
        model_pairs: List of (creator_model, critic_model) tuples to compare
        task_prompt: Task description for code generation
        config: Pattern configuration (uses defaults if None)

    Returns:
        Dictionary mapping model pair descriptions to their results
    """
    if model_pairs is None:
        # Default comparisons: different creator/critic combinations
        model_pairs = [
            ("gpt-4", "claude-sonnet-4-5-20250929"),
            ("gpt-4o", "claude-sonnet-4-5-20250929"),
            ("claude-sonnet-4-5-20250929", "gpt-4o"),
        ]

    config = config or ReflectionConfig()

    if task_prompt is None:
        task_prompt = config.task_prompt

    print(f"\n{'=' * 80}")
    print(f"Comparing Reflection Pattern Across Model Pairs")
    print(f"{'=' * 80}\n")

    results = {}
    for creator, critic in model_pairs:
        pair_name = f"{creator} (creator) + {critic} (critic)"
        print(f"\nTesting with {pair_name}...")
        try:
            result = run(
                creator_model=creator,
                critic_model=critic,
                task_prompt=task_prompt,
                config=config,
                verbose=False
            )
            results[pair_name] = result
            print(f"✓ Completed successfully ({result['iterations']} iterations)")
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[pair_name] = f"Error: {e}"

    # Write comparison results
    writer = create_writer("04_reflection")
    log_path = writer.write_comparison(
        results=results,
        input_data={"task_prompt": task_prompt}
    )

    print(f"\n{'=' * 80}")
    print(f"Comparison results written to: {log_path}")
    print(f"{'=' * 80}\n")

    return results


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    creator = sys.argv[1] if len(sys.argv) > 1 else None  # Uses default from config
    critic = sys.argv[2] if len(sys.argv) > 2 else None  # Uses default from config

    # Example tasks
    task1 = """Your task is to create a Python function named `calculate_factorial`.
This function should do the following:
1. Accept a single integer `n` as input.
2. Calculate its factorial (n!).
3. Include a clear docstring explaining what the function does.
4. Handle edge cases: The factorial of 0 is 1.
5. Handle invalid input: Raise a ValueError if the input is a negative number."""

    task2 = """Your task is to create a Python function named `fibonacci_sequence`.
This function should do the following:
1. Accept a single integer `n` as input.
2. Return a list containing the first `n` numbers of the Fibonacci sequence.
3. Include a clear docstring explaining what the function does.
4. Handle edge cases: n=0 should return an empty list, n=1 should return [0].
5. Handle invalid input: Raise a ValueError if the input is negative."""

    # Run with different tasks
    print("\n=== Running with task 1 (factorial) ===")
    run(creator_model=creator, critic_model=critic, task_prompt=task1)

    print("\n=== Running with task 2 (fibonacci) ===")
    run(creator_model=creator, critic_model=critic, task_prompt=task2)

    # Compare multiple model pairs (uncomment to use)
    # print("\n=== Comparing model pairs ===")
    # compare_models(model_pairs=[
    #     ("gpt-4o", "claude-sonnet-4-5-20250929"),
    #     ("gpt-4.1-2025-04-14", "claude-sonnet-4-5-20250929"),
    # ])
