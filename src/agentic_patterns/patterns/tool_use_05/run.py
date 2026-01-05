"""
Tool Use Pattern - Function calling with multiple specialized search tools.

This pattern demonstrates:
1. Defining multiple tools with specific domains (tech, science, history)
2. Agent selecting appropriate tool based on query content
3. Tool execution and response generation
4. LLM using tool results to formulate comprehensive answers
"""

import warnings
# Suppress Pydantic v1 deprecation warning for Python 3.14+
warnings.filterwarnings('ignore', message='.*Pydantic V1.*', category=UserWarning)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as langchain_tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agentic_patterns.common import ModelFactory, create_writer
from agentic_patterns.patterns.tool_use_05.config import ToolUseConfig


def run(
    model_name: str = "gpt-4o",
    queries: list[str] = None,
    config: ToolUseConfig = None,
    verbose: bool = True
):
    """
    Run the tool use pattern with a specified model and queries.

    This creates an agent that:
    1. Receives user queries
    2. Selects appropriate tool based on query domain
    3. Executes the tool to retrieve information
    4. Formulates comprehensive answer using tool results

    Args:
        model_name: Model to use for the agent (default: gpt-4o)
        queries: List of queries to process (uses defaults if None)
        config: Pattern configuration (uses defaults if None)
        verbose: Whether to print intermediate results

    Returns:
        Dictionary containing results for all queries
    """
    # Use provided config or defaults
    config = config or ToolUseConfig()

    # Default queries if none provided
    if queries is None:
        queries = [
            "What is Python programming language?",
            "Tell me about photosynthesis",
            "What was the Roman Empire?",
            "Explain quantum computing",
            "What is CRISPR technology?",
        ]

    # Initialize the Language Model
    llm = ModelFactory.create(model_name, **config.get_model_kwargs())

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Running Tool Use Pattern")
        print(f"Model: {model_name}")
        print(f"Number of queries: {len(queries)}")
        print(f"{'=' * 80}\n")

    # --- Define Tools ---
    # Each tool searches a specific domain of knowledge

    @langchain_tool
    def search_technology(query: str) -> str:
        """
        Search for information about technology, programming, and computing topics.
        Use this tool for queries about programming languages, software, hardware,
        quantum computing, AI, machine learning, blockchain, and other tech topics.

        Args:
            query: The technology-related search query

        Returns:
            Information about the technology topic
        """
        if verbose:
            print(f"🔧 TOOL CALLED: search_technology('{query}')")

        query_lower = query.lower()

        # Try to find a matching result
        for key, value in config.tech_search_results.items():
            if key in query_lower:
                return value

        return config.tech_search_results["default"]

    @langchain_tool
    def search_science(query: str) -> str:
        """
        Search for information about science, biology, physics, chemistry, and nature.
        Use this tool for queries about biological processes, scientific phenomena,
        natural sciences, medical topics, and scientific discoveries.

        Args:
            query: The science-related search query

        Returns:
            Information about the science topic
        """
        if verbose:
            print(f"🔬 TOOL CALLED: search_science('{query}')")

        query_lower = query.lower()

        # Try to find a matching result
        for key, value in config.science_search_results.items():
            if key in query_lower:
                return value

        return config.science_search_results["default"]

    @langchain_tool
    def search_history(query: str) -> str:
        """
        Search for information about historical events, civilizations, and periods.
        Use this tool for queries about ancient civilizations, wars, historical figures,
        cultural movements, and significant historical events.

        Args:
            query: The history-related search query

        Returns:
            Information about the historical topic
        """
        if verbose:
            print(f"📚 TOOL CALLED: search_history('{query}')")

        query_lower = query.lower()

        # Try to find a matching result
        for key, value in config.history_search_results.items():
            if key in query_lower:
                return value

        return config.history_search_results["default"]

    # Collect all tools
    tools = [search_technology, search_science, search_history]

    # Create a mapping of tool names to tool functions for execution
    tools_map = {tool.name: tool for tool in tools}

    # --- Bind Tools to LLM ---
    # This enables the LLM to call tools
    llm_with_tools = llm.bind_tools(tools)

    # --- Execute Queries ---
    results = {}

    for i, query in enumerate(queries, 1):
        if verbose:
            print(f"\n{'─' * 80}")
            print(f"Query {i}/{len(queries)}: {query}")
            print(f"{'─' * 80}")

        try:
            # Create messages for this query
            messages = [
                HumanMessage(content=f"{config.system_prompt}\n\nUser query: {query}")
            ]

            # Agent loop: allow multiple tool calls if needed
            max_iterations = 5
            for iteration in range(max_iterations):
                # Invoke LLM with tools
                response = llm_with_tools.invoke(messages)
                messages.append(response)

                # Check if LLM wants to call tools
                if not response.tool_calls:
                    # No tool calls, we have the final answer
                    output = response.content
                    break

                # Execute tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if verbose:
                        print(f"\n🔧 Iteration {iteration + 1}: LLM calling tool '{tool_name}'")

                    # Execute the tool
                    if tool_name in tools_map:
                        tool_result = tools_map[tool_name].invoke(tool_args)

                        # Add tool result to messages
                        messages.append(
                            ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call["id"]
                            )
                        )
                    else:
                        # Unknown tool
                        messages.append(
                            ToolMessage(
                                content=f"Error: Unknown tool '{tool_name}'",
                                tool_call_id=tool_call["id"]
                            )
                        )
            else:
                # Max iterations reached
                output = "Maximum iterations reached without final answer."

            results[query] = {
                "status": "success",
                "response": output
            }

            if verbose:
                print(f"\n✅ Response:")
                print(output)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            results[query] = {
                "status": "error",
                "response": error_msg
            }

            if verbose:
                print(f"\n❌ {error_msg}")

    # --- Summary ---
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"SUMMARY")
        print(f"{'=' * 80}")
        successful = sum(1 for r in results.values() if r["status"] == "success")
        print(f"Successful queries: {successful}/{len(queries)}")
        print(f"{'=' * 80}\n")

    # Write results to log file
    writer = create_writer("05_tool_use")
    log_path = writer.write_result(
        result={
            "queries": queries,
            "results": results,
            "summary": {
                "total_queries": len(queries),
                "successful": sum(1 for r in results.values() if r["status"] == "success"),
                "failed": sum(1 for r in results.values() if r["status"] == "error")
            }
        },
        model_name=model_name,
        input_data={"queries": queries},
        metadata={
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "num_tools": len(tools)
        }
    )

    if verbose:
        print(f"Results written to: {log_path}\n")

    return results


def compare_models(
    model_names: list[str] = None,
    queries: list[str] = None,
    config: ToolUseConfig = None
):
    """
    Compare the tool use pattern across multiple models.

    Args:
        model_names: List of model names to compare
        queries: List of queries to test (uses defaults if None)
        config: Pattern configuration (uses defaults if None)

    Returns:
        Dictionary mapping model names to their results
    """
    if model_names is None:
        model_names = [
            "gpt-4o",
            "gpt-4o-mini",
            "claude-sonnet-4-5-20250929",
        ]

    config = config or ToolUseConfig()

    if queries is None:
        queries = [
            "What is Python programming language?",
            "Tell me about photosynthesis",
            "What was the Roman Empire?",
        ]

    print(f"\n{'=' * 80}")
    print(f"Comparing Tool Use Pattern Across Models")
    print(f"{'=' * 80}\n")

    results = {}
    for model in model_names:
        print(f"\nTesting with {model}...")
        try:
            result = run(
                model_name=model,
                queries=queries,
                config=config,
                verbose=False
            )
            results[model] = result
            successful = sum(1 for r in result.values() if r["status"] == "success")
            print(f"✓ Completed successfully ({successful}/{len(queries)} queries answered)")
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[model] = f"Error: {e}"

    # Write comparison results
    writer = create_writer("05_tool_use")
    log_path = writer.write_comparison(
        results=results,
        input_data={"queries": queries}
    )

    print(f"\n{'=' * 80}")
    print(f"Comparison results written to: {log_path}")
    print(f"{'=' * 80}\n")

    return results


if __name__ == "__main__":
    import sys

    # Parse command line argument for model name
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o"

    # Example queries covering different domains
    example_queries = [
        # Technology queries
        "What is Python programming language?",
        "Explain machine learning",
        "What is blockchain technology?",

        # Science queries
        "Tell me about photosynthesis",
        "What is DNA?",
        "Explain black holes",

        # History queries
        "What was the Roman Empire?",
        "Tell me about the Renaissance",
        "What happened during World War II?",
    ]

    print(f"\n{'=' * 80}")
    print(f"Tool Use Pattern - Example Execution")
    print(f"{'=' * 80}\n")

    # Run with the specified model
    run(model_name=model, queries=example_queries)

    # Uncomment to run model comparison
    # print("\n\n=== Comparing Models ===")
    # compare_models()
