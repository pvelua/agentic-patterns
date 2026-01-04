"""
Chaining Pattern - Sequential LLM calls using LangChain Expression Language (LCEL).

This pattern demonstrates:
1. Creating prompt templates
2. Chaining LLM calls sequentially
3. Passing output from one step as input to the next
4. Using StrOutputParser to convert LLM output to strings
"""

import warnings
# Suppress Pydantic v1 deprecation warning for Python 3.14+
warnings.filterwarnings('ignore', message='.*Pydantic V1.*', category=UserWarning)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agentic_patterns.common import ModelFactory, create_writer
from agentic_patterns.patterns.chaining_01.config import ChainingConfig


def run(
    model_name: str = "gpt-4",
    input_text: str = None,
    config: ChainingConfig = None,
    verbose: bool = True
):
    """
    Run the chaining pattern with specified model.
    
    This creates a two-step chain:
    1. Extract technical specifications from text
    2. Transform specifications into JSON format
    
    Args:
        model_name: Model to use (e.g., 'gpt-4', 'claude-sonnet-4-5-20250929')
        input_text: Text to extract specifications from
        config: Pattern configuration (uses defaults if None)
        verbose: Whether to print intermediate results
        
    Returns:
        Final JSON output as a string
    """
    # Use provided config or defaults
    config = config or ChainingConfig()
    
    # Default input if none provided
    if input_text is None:
        input_text = (
            "The new laptop model features a 3.5 GHz octa-core processor, "
            "16GB of RAM, and a 1TB NVMe SSD."
        )
    
    # Initialize the Language Model
    llm = ModelFactory.create(model_name, **config.get_model_kwargs())
    
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Running Chaining Pattern with {model_name}")
        print(f"{'=' * 80}\n")
    
    # --- Step 1: Extract Information ---
    prompt_extract = ChatPromptTemplate.from_template(config.extraction_prompt)
    
    # --- Step 2: Transform to JSON ---
    prompt_transform = ChatPromptTemplate.from_template(config.transformation_prompt)
    
    # --- Build the Chain using LCEL ---
    # The StrOutputParser() converts the LLM's message output to a simple string
    extraction_chain = prompt_extract | llm | StrOutputParser()
    
    # The full chain passes the output of the extraction chain into the 'specifications'
    # variable for the transformation prompt
    full_chain = (
        {"specifications": extraction_chain}
        | prompt_transform
        | llm
        | StrOutputParser()
    )
    
    # --- Run the Chain ---
    if verbose:
        print(f"Input Text:\n{input_text}\n")
        print("Running extraction chain...")
    
    # Execute the chain with the input text dictionary
    final_result = full_chain.invoke({"text_input": input_text})
    
    if verbose:
        print(f"\n--- Final JSON Output ---")
        print(final_result)
        print(f"\n{'=' * 80}\n")
    
    # Write results to log file
    writer = create_writer("01_chaining")
    log_path = writer.write_result(
        result=final_result,
        model_name=model_name,
        input_data={"text_input": input_text},
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
    input_text: str = None,
    config: ChainingConfig = None
):
    """
    Compare the chaining pattern across multiple models.
    
    Args:
        models: List of model names to compare
        input_text: Text to extract specifications from
        config: Pattern configuration (uses defaults if None)
        
    Returns:
        Dictionary mapping model names to their results
    """
    if models is None:
        models = ["gpt-4", "gpt-3.5-turbo", "claude-sonnet-4-5-20250929"]
    
    config = config or ChainingConfig()
    
    if input_text is None:
        input_text = (
            "The new laptop model features a 3.5 GHz octa-core processor, "
            "16GB of RAM, and a 1TB NVMe SSD."
        )
    
    print(f"\n{'=' * 80}")
    print(f"Comparing Chaining Pattern Across Models")
    print(f"{'=' * 80}\n")
    
    results = {}
    for model in models:
        print(f"\nTesting with {model}...")
        try:
            result = run(
                model_name=model,
                input_text=input_text,
                config=config,
                verbose=False
            )
            results[model] = result
            print(f"✓ {model} completed successfully")
        except Exception as e:
            print(f"✗ {model} failed: {e}")
            results[model] = f"Error: {e}"
    
    # Write comparison results
    writer = create_writer("01_chaining")
    log_path = writer.write_comparison(
        results=results,
        input_data={"text_input": input_text}
    )
    
    print(f"\n{'=' * 80}")
    print(f"Comparison results written to: {log_path}")
    print(f"{'=' * 80}\n")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4"
    
    # Example custom input
    custom_input = (
        "The server configuration includes dual Intel Xeon processors at 2.8 GHz, "
        "128GB ECC RAM, and 4x 2TB SSDs in RAID 10."
    )
    
    # Run with default input
    print("\n=== Running with default input ===")
    run(model_name=model)
    
    # Run with custom input
    print("\n=== Running with custom input ===")
    run(model_name=model, input_text=custom_input)
    
    # Compare multiple models (uncomment to use)
    # print("\n=== Comparing models ===")
    # compare_models(models=["gpt-4", "gpt-3.5-turbo"])