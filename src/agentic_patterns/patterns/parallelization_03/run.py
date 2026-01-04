"""
Parallelization Pattern - Concurrent execution of multiple LLM chains.

This pattern demonstrates:
1. Executing multiple independent LLM chains in parallel
2. Using RunnableParallel for concurrent execution
3. Synthesizing results from multiple parallel operations
4. Reducing overall execution time through parallelization
"""

import warnings
# Suppress Pydantic v1 deprecation warning for Python 3.14+
warnings.filterwarnings('ignore', message='.*Pydantic V1.*', category=UserWarning)

import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from agentic_patterns.common import ModelFactory, create_writer
from agentic_patterns.patterns.parallelization_03.config import ParallelizationConfig


async def run(
    model_name: str = "gpt-4",
    topic: str = None,
    config: ParallelizationConfig = None,
    verbose: bool = True
):
    """
    Run the parallelization pattern with specified model.

    This creates three parallel chains that:
    1. Summarize the topic
    2. Generate questions about the topic
    3. Extract key terms from the topic
    4. Synthesize all results into a comprehensive answer

    Args:
        model_name: Model to use (e.g., 'gpt-4', 'claude-sonnet-4-5-20250929')
        topic: Topic to analyze
        config: Pattern configuration (uses defaults if None)
        verbose: Whether to print intermediate results

    Returns:
        Synthesized response as a string
    """
    # Use provided config or defaults
    config = config or ParallelizationConfig()

    # Default topic if none provided
    if topic is None:
        topic = "The history of space exploration"

    # Initialize the Language Model
    llm = ModelFactory.create(model_name, **config.get_model_kwargs())

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Running Parallelization Pattern with {model_name}")
        print(f"{'=' * 80}\n")

    # --- Define Independent Chains ---
    # These three chains represent distinct tasks that can be executed in parallel

    summarize_chain = (
        ChatPromptTemplate.from_messages([
            ("system", config.summarize_prompt),
            ("user", "{topic}")
        ])
        | llm
        | StrOutputParser()
    )

    questions_chain = (
        ChatPromptTemplate.from_messages([
            ("system", config.questions_prompt),
            ("user", "{topic}")
        ])
        | llm
        | StrOutputParser()
    )

    terms_chain = (
        ChatPromptTemplate.from_messages([
            ("system", config.terms_prompt),
            ("user", "{topic}")
        ])
        | llm
        | StrOutputParser()
    )

    # --- Build the Parallel + Synthesis Chain ---

    # Define the block of tasks to run in parallel
    map_chain = RunnableParallel(
        {
            "summary": summarize_chain,
            "questions": questions_chain,
            "key_terms": terms_chain,
            "topic": RunnablePassthrough(),  # Pass the original topic through
        }
    )

    # Define the final synthesis prompt which will combine the parallel results
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", config.synthesis_prompt),
        ("user", "Original topic: {topic}")
    ])

    # Construct the full chain by piping the parallel results into synthesis
    full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()

    # --- Run the Parallel Chain ---
    if verbose:
        print(f"Topic: {topic}\n")
        print("Running parallel analysis (summarize, questions, key terms)...")

    # Execute the chain asynchronously
    final_result = await full_parallel_chain.ainvoke(topic)

    if verbose:
        print(f"\n--- Synthesized Response ---")
        print(final_result)
        print(f"\n{'=' * 80}\n")

    # Write results to log file
    writer = create_writer("03_parallelization")
    log_path = writer.write_result(
        result=final_result,
        model_name=model_name,
        input_data={"topic": topic},
        metadata={
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }
    )

    if verbose:
        print(f"Results written to: {log_path}\n")

    return final_result


async def compare_models(
    models: list[str] = None,
    topic: str = None,
    config: ParallelizationConfig = None
):
    """
    Compare the parallelization pattern across multiple models.

    Args:
        models: List of model names to compare
        topic: Topic to analyze
        config: Pattern configuration (uses defaults if None)

    Returns:
        Dictionary mapping model names to their results
    """
    if models is None:
        models = ["gpt-4", "gpt-4o", "claude-sonnet-4-5-20250929"]

    config = config or ParallelizationConfig()

    if topic is None:
        topic = "The history of space exploration"

    print(f"\n{'=' * 80}")
    print(f"Comparing Parallelization Pattern Across Models")
    print(f"{'=' * 80}\n")

    results = {}
    for model in models:
        print(f"\nTesting with {model}...")
        try:
            result = await run(
                model_name=model,
                topic=topic,
                config=config,
                verbose=False
            )
            results[model] = result
            print(f"✓ {model} completed successfully")
        except Exception as e:
            print(f"✗ {model} failed: {e}")
            results[model] = f"Error: {e}"

    # Write comparison results
    writer = create_writer("03_parallelization")
    log_path = writer.write_comparison(
        results=results,
        input_data={"topic": topic}
    )

    print(f"\n{'=' * 80}")
    print(f"Comparison results written to: {log_path}")
    print(f"{'=' * 80}\n")

    return results


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4"

    # Example topics
    topic1 = "The history of space exploration"
    topic2 = "Artificial intelligence and machine learning"
    topic3 = "Climate change and renewable energy"

    # Run with different topics
    print("\n=== Running with topic 1 ===")
    asyncio.run(run(model_name=model, topic=topic1))

    print("\n=== Running with topic 2 ===")
    asyncio.run(run(model_name=model, topic=topic2))

    print("\n=== Running with topic 3 ===")
    asyncio.run(run(model_name=model, topic=topic3))

    # Compare multiple models (uncomment to use)
    # print("\n=== Comparing models ===")
    # asyncio.run(compare_models(models=["gpt-4o", "claude-sonnet-4-5-20250929"]))
