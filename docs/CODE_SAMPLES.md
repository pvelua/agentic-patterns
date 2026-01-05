# Pattern 1 - Request Chaining

```python
import os 
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 

# For better security, load environment variables from a .env file 
# from dotenv import load_dotenv # load_dotenv() 
# Make sure your OPENAI_API_KEY is set in the .env file 
# Initialize the Language Model (using ChatOpenAI is recommended) 

llm = ChatOpenAI(temperature=0) 
# --- Prompt 1: Extract Information --- 
prompt_extract = ChatPromptTemplate.from_template("Extract the technical specifications from the following text:\n\n{text_input}" ) 

# --- Prompt 2: Transform to JSON --- 
prompt_transform = ChatPromptTemplate.from_template("Transform the following specifications into a JSON object with 'cpu', 'memory', and 'storage' as keys:\n\n{specifications}" ) 

# --- Build the Chain using LCEL --- 
# The StrOutputParser() converts the LLM's message output to a simple string. 
extraction_chain = prompt_extract | llm | StrOutputParser() 

# The full chain passes the output of the extraction chain into the 'specifications' 
# variable for the transformation prompt. 
full_chain = ( 
	{"specifications": extraction_chain} 
	| prompt_transform 
	| llm 
	| StrOutputParser() 
) 

# --- Run the Chain --- 
input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD." 

# Execute the chain with the input text dictionary. 

final_result = full_chain.invoke({"text_input": input_text}) 
print("\n--- Final JSON Output ---") 
print(final_result)
```

# Pattern 2 - Routing

## Code exaple (LangChain)

This code demonstrates a simple agent-like system using LangChain and Google’s Generative AI. It sets up a “coordinator” that routes user requests to different simulated “sub-agent” handlers based on the request’s intent (booking, information, or unclear). The system uses a language model to classify the request and then delegates it to the appropriate handler function, simulating a basic delegation pattern often seen in multi-agent architectures.

```python
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough, RunnableBranch 
# --- Configuration --- 
# Ensure your API key environment variable is set (e.g., GOOGLE_API_KEY) 
try:    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)    
    print(f"Language model initialized: {llm.model}") 
except Exception as e:    
    print(f"Error initializing language model: {e}")    
    llm = None 

# --- Define Simulated Sub-Agent Handlers (equivalent to ADK sub_agents) --- 
def booking_handler(request: str) -> str:
    """Simulates the Booking Agent handling a request."""
    print("\n--- DELEGATING TO BOOKING HANDLER ---")    
    return f"Booking Handler processed request: '{request}'. Result: Simulated booking action." 
    
def info_handler(request: str) -> str:
    """Simulates the Info Agent handling a request."""    
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return f"Info Handler processed request: '{request}'. Result: Simulated information retrieval."
    
def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be delegated."""
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return f"Coordinator could not delegate request: '{request}'. Please clarify." 
    
# --- Define Coordinator Router Chain (equivalent to ADK coordinator's instruction) --- 
# This chain decides which handler to delegate to. 
coordinator_router_prompt = ChatPromptTemplate.from_messages([    
    ("system", """Analyze the user's request and determine which specialist handler should process it.    - If the request is related to booking flights or hotels, output 'booker'.
       - For all other general information questions, output 'info'.
       - If the request is unclear or doesn't fit either category, output 'unclear'.
      ONLY output one word: 'booker', 'info', or 'unclear'."""),
    ("user", "{request}") 
 ]) 
 if llm:
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()
    
# --- Define the Delegation Logic (equivalent to ADK's Auto-Flow based on sub_agents) --- 
# Use RunnableBranch to route based on the router chain's output. 
# Define the branches for the RunnableBranch 
branches = {
    "booker": RunnablePassthrough.assign(output=lambda x: booking_handler(x['request']['request'])), "info": RunnablePassthrough.assign(output=lambda x: info_handler(x['request']['request'])), "unclear": RunnablePassthrough.assign(output=lambda x: unclear_handler(x['request']['request'])), 
}

# Create the RunnableBranch. It takes the output of the router chain
# and routes the original input ('request') to the corresponding handler. 
delegation_branch = RunnableBranch(
    (lambda x: x['decision'].strip() == 'booker', branches["booker"]), # Added .strip()    
    (lambda x: x['decision'].strip() == 'info', branches["info"]),     # Added .strip()    
    branches["unclear"] # Default branch for 'unclear' or any other output 
    )
# Combine the router chain and the delegation branch into a single runnable
# The router chain's output ('decision') is passed along with the original input ('request')
# # to the delegation_branch.
coordinator_agent = {
    "decision": coordinator_router_chain,    
    "request": RunnablePassthrough() 
} | delegation_branch | (lambda x: x['output'])  # Extract the final output 

# --- Example Usage --- 
def main():    
    if not llm:
        print("\nSkipping execution due to LLM initialization failure.")
        return
        
    print("--- Running with a booking request ---")    
    request_a = "Book me a flight to London."    
    result_a = coordinator_agent.invoke({"request": request_a})    
    print(f"Final Result A: {result_a}")    
    
    print("\n--- Running with an info request ---")    
    request_b = "What is the capital of Italy?" 
    result_b = coordinator_agent.invoke({"request": request_b})    
    print(f"Final Result B: {result_b}")    
    
    print("\n--- Running with an unclear request ---")    
    request_c = "Tell me about quantum physics."    
    result_c = coordinator_agent.invoke({"request": request_c})    
    print(f"Final Result C: {result_c}") 
    
if __name__ == "__main__":
    main()

```

## Code example (Google ADK)

This Python code demonstrates an example of an Agent Development Kit (ADK) application using Google’s ADK library. It sets up a “Coordinator” agent that routes user requests to specialized sub-agents (“Booker” for bookings and “Info” for general information) based on defined instructions. The sub-agents then use specific tools to simulate handling the requests, showcasing a basic delegation pattern within an agent system.

```python
# Copyright (c) 2025 Marco Fago # # This code is licensed under the MIT License. # See the LICENSE file in the repository for the full license text.
import uuid 
from typing import Dict, Any, Optional 
from google.adk.agents import Agent 
from google.adk.runners import InMemoryRunner 
from google.adk.tools import FunctionTool 
from google.genai import types 
from google.adk.events import Event 

# --- Define Tool Functions --- 
# These functions simulate the actions of the specialist agents. 

def booking_handler(request: str) -> str:
    """
    Handles booking requests for flights and hotels.    
    Args: 
        request: The user's request for a booking.    
    Returns:
        A confirmation message that the booking was handled.
    """
    print("------------- Booking Handler Called -------------")
    return f"Booking action for '{request}' has been simulated."

def info_handler(request: str) -> str:
    """
    Handles general information requests.
    Args:
        request: The user's question.
    Returns:
        A message indicating the information request was handled.
    """    
    print("------------- Info Handler Called ----------------")
    return f"Information request for '{request}'. Result: Simulated information retrieval."
    
def unclear_handler(request: str) -> str:
    """
    Handles requests that couldn't be delegated.
    """    
    return f"Coordinator could not delegate request: '{request}'. Please clarify."
    
# --- Create Tools from Functions --- 
booking_tool = FunctionTool(booking_handler) 
info_tool = FunctionTool(info_handler)

# Define specialized sub-agents equipped with their respective tools 
booking_agent = Agent(
    name="Booker",
    model="gemini-2.0-flash",
    description="A specialized agent that handles all flight
                and hotel booking requests by calling the booking tool.",
    tools=[booking_tool] ) 

info_agent = Agent(
    name="Info",    
    model="gemini-2.0-flash",
    description="A specialized agent that provides general information
               and answers user questions by calling the info tool.",
    tools=[info_tool] )
    
# Define the parent agent with explicit delegation instructions 
coordinator = Agent(
    name="Coordinator",
    model="gemini-2.0-flash",
    instruction=(
        "You are the main coordinator. Your only task is to analyze incoming user requests "
        "and delegate them to the appropriate specialist agent. Do not try to answer the user directly.\n"
        "- For any requests related to booking flights or hotels,delegate to the 'Booker' agent.\n"
        "- For all other general information questions, delegate to the 'Info' agent."    
        ),
    description="A coordinator that routes user requests to the correct specialist agent.",    
    # The presence of sub_agents enables LLM-driven delegation (Auto-Flow) by default.
    sub_agents=[booking_agent, info_agent] )
    
# --- Execution Logic --- 
async def run_coordinator(runner: InMemoryRunner, request: str):
    """Runs the coordinator agent with a given request and delegates."""
    print(f"\n--- Running Coordinator with request: '{request}' ---")    
    final_result = ""    
    try:        
        user_id = "user_123"
        session_id = str(uuid.uuid4())        
        await runner.session_service.create_session(            
            app_name=runner.app_name, user_id=user_id, session_id=session_id        
        )        
        
        for event in runner.run(           
            user_id=user_id,            
            session_id=session_id,            
            new_message=types.Content(                
                role='user',                
                parts=[types.Part(text=request)]            
            ),        
        ):            
            if event.is_final_response() and event.content:
                # Try to get text directly from event.content
                # to avoid iterating parts
                if hasattr(event.content, 'text') and event.content.text:
                    final_result = event.content.text
                elif event.content.parts:
                    # Fallback: Iterate through parts and extract text (might trigger warning)
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)
                # Assuming the loop should break after the final response
                break
            
        print(f"Coordinator Final Response: {final_result}")
        return final_result
    except Exception as e:
        print(f"An error occurred while processing your request: {e}")
        return f"An error occurred while processing your request: {e}"

async def main():
    """Main function to run the ADK example."""
    print("--- Google ADK Routing Example (ADK Auto-Flow Style) ---")
    print("Note: This requires Google ADK installed and authenticated.")
    
    runner = InMemoryRunner(coordinator)
    # Example Usage    
    result_a = await run_coordinator(runner, "Book me a hotel in Paris.")
    print(f"Final Output A: {result_a}")
    result_b = await run_coordinator(runner, "What is the highest mountain in the world?")
    print(f"Final Output B: {result_b}")
    result_c = await run_coordinator(runner, "Tell me a random fact.") # Should go to Info
    print(f"Final Output C: {result_c}")
    result_d = await run_coordinator(runner, "Find flights to Tokyo next month.") # Should go to Booker
    print(f"Final Output D: {result_d}") 

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()    
    await main()

```

# Pattern 3 - Parallelization

Parallelization involves executing multiple components, such as LLM calls, tool usages, or even entire sub-agents, concurrently. Instead of waiting for one step to complete before starting the next, parallel execution allows independent tasks to run at the same time, significantly reducing the overall execution time for tasks that can be broken down into independent parts. Typical use cases:
- Information gathering and research
- Data processing and analysis
- Multi-API or tool interaction / use
- Content generation with multiple components
- Validation and verification
- Multi-modal processing
- A/ B testing pr multiple oprions generation

## Code Example (LangChain)

Parallel execution within the LangChain framework is facilitated by the LangChain Expression Language (LCEL). The primary method involves structuring multiple runnable components within a dictionary or list construct. When this collection is passed as input to a subsequent component in the chain, the LCEL runtime executes the contained runnables concurrently. 
```python
import os
import asyncio

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

# --- Configuration --- 
# Ensure your API key environment variable is set (e.g., OPENAI_API_KEY)
try:
    llm: Optional[ChatOpenAI] = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None 
    
# --- Define Independent Chains --- 
# These three chains represent distinct tasks that can be executed in parallel. 

summarize_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarize the following topic concisely:"),        
        ("user", "{topic}")    
    ])
    | llm
    | StrOutputParser() 
) 

questions_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Generate three interesting questions about the following topic:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser() 
)

terms_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Identify 5-10 key terms from the following topic, separated by commas:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

# --- Build the Parallel + Synthesis Chain --- 

# 1. Define the block of tasks to run in parallel. The results of these,
#    along with the original topic, will be fed into the next step.
map_chain = RunnableParallel(
    {
        "summary": summarize_chain,
        "questions": questions_chain,
        "key_terms": terms_chain,
        "topic": RunnablePassthrough(), # Pass the original topic through
    } 
)

# 2. Define the final synthesis prompt which will combine the parallel results.
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """Based on the following information:
    Summary: {summary}
    Related Questions: {questions}
    Key Terms: {key_terms}
    Synthesize a comprehensive answer."""),
    ("user", "Original topic: {topic}")
])

# 3. Construct the full chain by piping the parallel results directly
#    into the synthesis prompt, followed by the LLM and output parser. 
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()

# --- Run the Chain --- 
async def run_parallel_example(topic: str) -> None:
    """
    Asynchronously invokes the parallel processing chain with a specific topic
    and prints the synthesized result.
    
    Args:
        topic: The input topic to be processed by the LangChain chains.
    """
    if not llm:
        print("LLM not initialized. Cannot run example.")
        return
    
    print(f"\n--- Running Parallel LangChain Example for Topic: '{topic}' ---")
    try:
        # The input to 'ainvoke' is the single 'topic' string,
        # then passed to each runnable in the 'map_chain'.
        response = await full_parallel_chain.ainvoke(topic)
        print("\n--- Final Response ---")
        print(response)
    except Exception as e:
        print(f"\nAn error occurred during chain execution: {e}")
        
 if __name__ == "__main__":
    test_topic = "The history of space exploration"
    # In Python 3.7+, asyncio.run is the standard way to run an async function.
    asyncio.run(run_parallel_example(test_topic))
```
## Code Example (Google ADK)

The sample sets up three LlmAgent instances to act as specialized researchers. ResearcherAgent_1 focuses on renewable energy sources, ResearcherAgent_2 researches electric vehicle technology, and ResearcherAgent_3 investigates carbon capture methods. Each researcher agent is configured to use a GEMINI_MODEL and the google_search tool. They are instructed to summarize their findings concisely (1–2 sentences) and store these summaries in the session state using output_key. A ParallelAgent named ParallelWebResearchAgent is then created to run these three reseaxher agents concurrently.

```python
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent 
from google.adk.tools import google_search 

GEMINI_MODEL="gemini-2.0-flash" 

# --- 1. Define Researcher Sub-Agents (to run in parallel) --- 
# Researcher 1: Renewable Energy 
researcher_agent_1 = LlmAgent(
    name="RenewableEnergyResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant specializing in energy. Research the latest advancements in 'renewable energy sources'. Use the Google Search tool provided. Summarize your key findings concisely (1-2 sentences). Output *only* the summary. """,
    description="Researches renewable energy sources.",
    tools=[google_search],
    # Store result in state for the merger agent
    output_key="renewable_energy_result" 
) 

# Researcher 2: Electric Vehicles 
researcher_agent_2 = LlmAgent(
    name="EVResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant specializing in transportation. Research the latest developments in 'electric vehicle technology'. Use the Google Search tool provided. Summarize your key findings concisely (1-2 sentences). Output *only* the summary. """,
    description="Researches electric vehicle technology.",
    tools=[google_search],
    # Store result in state for the merger agent
    output_key="ev_technology_result" 
)

# Researcher 3: Carbon Capture researcher_agent_3 = LlmAgent(     name="CarbonCaptureResearcher",     model=GEMINI_MODEL,     instruction="""You are an AI Research Assistant specializing in climate solutions. Research the current state of 'carbon capture methods'. Use the Google Search tool provided. Summarize your key findings concisely (1-2 sentences). Output *only* the summary. """,     description="Researches carbon capture methods.",     tools=[google_search],     # Store result in state for the merger agent     output_key="carbon_capture_result" ) # --- 2. Create the ParallelAgent (Runs researchers concurrently) --- # This agent orchestrates the concurrent execution of the researchers. # It finishes once all researchers have completed and stored their results in state. parallel_research_agent = ParallelAgent(     name="ParallelWebResearchAgent",     sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],     description="Runs multiple research agents in parallel to gather information." ) # --- 3. Define the Merger Agent (Runs *after* the parallel agents) --- # This agent takes the results stored in the session state by the parallel agents # and synthesizes them into a single, structured response with attributions. merger_agent = LlmAgent(     name="SynthesisAgent",     model=GEMINI_MODEL, # Or potentially a more powerful model if needed for synthesis     instruction="""You are an AI Assistant responsible for combining research findings into a structured report. Your primary task is to synthesize the following research summaries, clearly attributing findings to their source areas. Structure your response using headings for each topic. Ensure the report is coherent and integrates the key points smoothly. **Crucially: Your entire response MUST be grounded *exclusively* on the information provided in the 'Input Summaries' below. Do NOT add any external knowledge, facts, or details not present in these specific summaries.** **Input Summaries:** *   **Renewable Energy:**     {renewable_energy_result} *   **Electric Vehicles:**     {ev_technology_result} *   **Carbon Capture:**     {carbon_capture_result} 

**Output Format:** 

## Summary of Recent Sustainable Technology Advancements 

### Renewable Energy Findings 
(Based on RenewableEnergyResearcher's findings) 
[Synthesize and elaborate *only* on the renewable energy input summary provided above.] 

### Electric Vehicle Findings 
(Based on EVResearcher's findings) 
[Synthesize and elaborate *only* on the EV input summary provided above.] 

### Carbon Capture Findings 
(Based on CarbonCaptureResearcher's findings) 
[Synthesize and elaborate *only* on the carbon capture input summary provided above.] 

### Overall Conclusion 
[Provide a brief (1-2 sentence) concluding statement that connects *only* the findings presented above.] 

Output *only* the structured report following this format. Do not include introductory or concluding phrases outside this structure, and strictly adhere to using only the provided input summary content. 
""",     
    description="Combines research findings from parallel agents into a structured, cited report, strictly grounded on provided inputs.",     
    # No tools needed for merging     
    # No output_key needed here, as its direct response is the final output of the sequence 
) 

# --- 4. Create the SequentialAgent (Orchestrates the overall flow) --- 
# This is the main agent that will be run. It first executes the ParallelAgent 
# to populate the state, and then executes the MergerAgent to produce the final output. 
sequential_pipeline_agent = SequentialAgent(
    name="ResearchAndSynthesisPipeline",
    # Run parallel research first, then merge
    sub_agents=[parallel_research_agent, merger_agent],
    description="Coordinates parallel research and synthesizes the results." 
) 
root_agent = sequential_pipeline_agent
```

# Pattern 4 - Reflection

The **Reflection pattern** involves an agent evaluating its own work, output, or internal state and using that evaluation to improve its performance or refine its response. It’s a form of self-correction or self-improvement, allowing the agent to iteratively refine its output or adjust its approach based on feedback, internal critique, or comparison against desired criteria. Reflection can occasionally be facilitated by a separate agent whose specific role is to analyze the output of an initial agent. The process typically involves:
1. Execution (agent in Producer role)
2. Evaluation / Critique (agent in Critic role)
3. Reflection / Refinement (agent in Producer role)
4. Iteration (optional but common)

Typical use cases:
- Creative writing and content generation
- Code generation and debugging
- Complex problem solving
- Summarization and information Synthesis
- Planning and strategy
- Conversational agents (e.g., in customer support scenarios)

## Code Example (LangChain)

```python
import os from dotenv
import load_dotenv from langchain_openai
import ChatOpenAI from langchain_core.prompts
import ChatPromptTemplate from langchain_core.messages
import SystemMessage, HumanMessage

 # --- Configuration --- 
 # Load environment variables from .env file (for OPENAI_API_KEY) 
 load_dotenv()

 # Check if the API key is set
 if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file. Please add it.")
    
# Initialize the Chat LLM. We use gpt-4o for better reasoning.
# A lower temperature is used for more deterministic outputs.
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

def run_reflection_loop():
    """    Demonstrates a multi-step AI reflection loop to progressively improve a Python function.    """
    # --- The Core Task ---
    task_prompt = """
    Your task is to create a Python function named `calculate_factorial`.
    This function should do the following:
    1. Accept a single integer `n` as input.
    2. Calculate its factorial (n!).
    3. Include a clear docstring explaining what the function does.
    4. Handle edge cases: The factorial of 0 is 1.
    5. Handle invalid input: Raise a ValueError if the input is a negative number.
    """
    # --- The Reflection Loop ---
    max_iterations = 3
    current_code = ""
    # We will build a conversation history to provide context in each step.
    message_history = [HumanMessage(content=task_prompt)]
    
    for i in range(max_iterations):
        print("\n" + "="*25 + f" REFLECTION LOOP: ITERATION {i + 1} " + "="*25)
        
        # --- 1. GENERATE / REFINE STAGE ---
        # In the first iteration, it generates. In subsequent iterations, it refines.
        if i == 0:
            print("\n>>> STAGE 1: GENERATING initial code...")
            # The first message is just the task prompt.
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            print("\n>>> STAGE 1: REFINING code based on previous critique...")
            # The message history now contains the task,
            # the last code, and the last critique.
            # We instruct the model to apply the critiques.
            message_history.append(HumanMessage(content="Please refine the code using the critiques provided."))
            response = llm.invoke(message_history)
            current_code = response.content
            
        print("\n--- Generated Code (v" + str(i + 1) + ") ---\n" + current_code)
        message_history.append(response) # Add the generated code to history
        
        # --- 2. REFLECT STAGE ---
        print("\n>>> STAGE 2: REFLECTING on the generated code...")
        
        # Create a specific prompt for the reflector agent.
        # This asks the model to act as a senior code reviewer.
        reflector_prompt = [
            SystemMessage(content="""
                You are a senior software engineer and an expert in Python.
                Your role is to perform a meticulous code review.
                Critically evaluate the provided Python code based on the original task requirements.
                Look for bugs, style issues, missing edge cases, and areas for improvement.
                If the code is perfect and meets all requirements, respond with the single phrase CODE_IS_PERFECT'.
                Otherwise, provide a bulleted list of your critiques.
            """),
            HumanMessage(content=f"Original Task:\n{task_prompt}\n\nCode to Review:\n{current_code}")        
        ]
        critique_response = llm.invoke(reflector_prompt)
        critique = critique_response.content
        
        # --- 3. STOPPING CONDITION ---
        if "CODE_IS_PERFECT" in critique:
            print("\n--- Critique ---\nNo further critiques found. The code is satisfactory.")
            break
        
        print("\n--- Critique ---\n" + critique)
        # Add the critique to the history for the next refinement loop.
        message_history.append(HumanMessage(content=f"Critique of the previous code:\n{critique}"))

    print("\n" + "="*30 + " FINAL RESULT " + "="*30)
    print("\nFinal refined code after the reflection process:\n")
    print(current_code)
        
 if __name__ == "__main__":    run_reflection_loop()

```

## Code Example (Google ADK)
```python
from google.adk.agents import SequentialAgent, LlmAgent 

# The first agent generates the initial draft. 
generator = LlmAgent(
    name="DraftWriter",
    description="Generates initial draft content on a given subject.",
    instruction="Write a short, informative paragraph about the user's subject.",
    output_key="draft_text" # The output is saved to this state key. 
)

# The second agent critiques the draft from the first agent. 
reviewer = LlmAgent(
    name="FactChecker",
    description="Reviews a given text for factual accuracy and provides a structured critique.",
    instruction="""
        You are a meticulous fact-checker.
        1. Read the text provided in the state key 'draft_text'.
        2. Carefully verify the factual accuracy of all claims.
        3. Your final output must be a dictionary containing two keys:
            - "status": A string, either "ACCURATE" or "INACCURATE".
            - "reasoning": A string providing a clear explanation for your status, citing specific issues if any are found.
    """,
    output_key="review_output" # The structured dictionary is saved here. 
)

# The SequentialAgent ensures the generator runs before the reviewer.
review_pipeline = SequentialAgent(
    name="WriteAndReview_Pipeline",
    sub_agents=[generator, reviewer] 
)

# Execution Flow: 
# 1. generator runs -> saves its paragraph to state['draft_text']. 
# 2. reviewer runs -> reads state['draft_text'] and saves its dictionary output to state['review_output'].

```

# Pattern 5 - Tool Use (Function Calling)

## Code Example (LangChain)

The implementation of tool use within the LangChain framework is a two-stage process. Initially, one or more tools are defined, typically by encapsulating existing Python functions or other runnable components. Subsequently, these tools are bound to a language model,

```python
import os, getpass
import asyncio
import nest_asyncio from typing
import List from dotenv
import load_dotenv
import logging from langchain_google_genai
import ChatGoogleGenerativeAI from langchain_core.prompts
import ChatPromptTemplate from langchain_core.tools
import tool as langchain_tool from langchain.agents
import create_tool_calling_agent, AgentExecutor

# UNCOMMENT
# Prompt the user securely and set API keys as an environment variables
# os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")
# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

try:
    # A model with function/tool calling capabilities is required.
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    print(f"✅ Language model initialized: {llm.model}")
except Exception as e:
     print(f"🛑 Error initializing language model: {e}")
    llm = None
    
# --- Define a Tool ---
@langchain_tool
def search_information(query: str) -> str:
    """ Provides factual information on a given topic. Use this tool to find answers to phrases
        like 'capital of France' or 'weather in London?'.
    """
    print(f"\n--- 🛠 Tool Called: search_information with query: '{query}' ---")
    # Simulate a search tool with a dictionary of predefined results.
    simulated_results = {
        "weather in london": "The weather in London is currently cloudy with a temperature of 15°C.",
        "capital of france": "The capital of France is Paris.",
        "population of earth": "The estimated population of Earth is around 8 billion people.",
        "tallest mountain": "Mount Everest is the tallest mountain above sea level.",
        "default": f"Simulated search result for '{query}': No specific information found, but the topic seems interesting."
    }
    result = simulated_results.get(query.lower(), simulated_results["default"])
    print(f"--- TOOL RESULT: {result} ---")
    return result
    
tools = [search_information]

# --- Create a Tool-Calling Agent ---
if llm:
    # This prompt template requires an `agent_scratchpad` placeholder for the agent's internal steps.
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create the agent, binding the LLM, tools, and prompt together.
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    
    # AgentExecutor is the runtime that invokes the agent and executes the chosen tools.
    # The 'tools' argument is not needed here as they are already bound to the agent.
    agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools) 

async def run_agent_with_tool(query: str):
    """Invokes the agent executor with a query and prints the final response."""
    print(f"\n--- 🏃 Running Agent with Query: '{query}' ---")
    try:
        response = await agent_executor.ainvoke({"input": query})
        print("\n--- ✅ Final Agent Response ---")
        print(response["output"])   
    except Exception as e:
        print(f"\n🛑 An error occurred during agent execution: {e}") 

async def main():
    """Runs all agent queries concurrently."""
    tasks = [
        run_agent_with_tool("What is the capital of France?"),
        run_agent_with_tool("What's the weather like in London?"),
        run_agent_with_tool("Tell me something about dogs.") # Should trigger the default tool response
    ]

await asyncio.gather(*tasks)
nest_asyncio.apply()
asyncio.run(main())
```

## Code Example (Crew AI)

```python
# pip install crewai langchain-openai
import os from crewai
import Agent, Task, Crew from crewai.tools
import tool import logging

# --- Best Practice: Configure Logging ---
# A basic logging setup helps in debugging and tracking the crew's execution.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Set up your API Key ---
# For production, it's recommended to use a more secure method for key management
# like environment variables loaded at runtime or a secret manager.
#
# Set the environment variable for your chosen LLM provider (e.g., OPENAI_API_KEY)
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
# os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"

# --- 1. Refactored Tool: Returns Clean Data ---
# The tool now returns raw data (a float) or raises a standard Python error.
# This makes it more reusable and forces the agent to handle outcomes properly.
@tool("Stock Price Lookup Tool")
def get_stock_price(ticker: str) -> float:
    """
    Fetches the latest simulated stock price for a given stock ticker symbol.
    Returns the price as a float. Raises a ValueError if the ticker is not found.
    """
    logging.info(f"Tool Call: get_stock_price for ticker '{ticker}'")
    simulated_prices = {        "AAPL": 178.15,        "GOOGL": 1750.30,        "MSFT": 425.50,    }    price = simulated_prices.get(ticker.upper())    if price is not None:        return price    else:        # Raising a specific error is better than returning a string.        # The agent is equipped to handle exceptions and can decide on the next action.        raise ValueError(f"Simulated price for ticker '{ticker.upper()}' not found.") # --- 2. Define the Agent --- # The agent definition remains the same, but it will now leverage the improved tool. financial_analyst_agent = Agent(  role='Senior Financial Analyst',  goal='Analyze stock data using provided tools and report key prices.',  backstory="You are an experienced financial analyst adept at using data sources to find stock information. You provide clear, direct answers.",  verbose=True,  tools=[get_stock_price],  # Allowing delegation can be useful, but is not necessary for this simple task.  allow_delegation=False, ) # --- 3. Refined Task: Clearer Instructions and Error Handling --- # The task description is more specific and guides the agent on how to react # to both successful data retrieval and potential errors. analyze_aapl_task = Task(  description=(      "What is the current simulated stock price for Apple (ticker: AAPL)? "      "Use the 'Stock Price Lookup Tool' to find it. "      "If the ticker is not found, you must report that you were unable to retrieve the price."  ),  expected_output=(      "A single, clear sentence stating the simulated stock price for AAPL. "      "For example: 'The simulated stock price for AAPL is $178.15.' "      "If the price cannot be found, state that clearly."  ),  agent=financial_analyst_agent, ) # --- 4. Formulate the Crew --- # The crew orchestrates how the agent and task work together. financial_crew = Crew(  agents=[financial_analyst_agent],  tasks=[analyze_aapl_task],  verbose=True # Set to False for less detailed logs in production ) # --- 5. Run the Crew within a Main Execution Block --- # Using a __name__ == "__main__": block is a standard Python best practice. 

def main():
    """Main function to run the crew."""
    # Check for API key before starting to avoid runtime errors.
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: The OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
        return
        
    print("\n## Starting the Financial Crew...")
    print("---------------------------------")
    # The kickoff method starts the execution.
    result = financial_crew.kickoff()
    
    print("\n---------------------------------")
    print("## Crew execution finished.")
    print("\nFinal Result:\n", result) 

if __name__ == "__main__":
    main()
```

## Code Example (Goole ADK)

```python
from google.adk.agents
import Agent from google.adk.runners
import Runner from google.adk.sessions
import InMemorySessionService from google.adk.tools
import google_search from google.genai
import types import nest_asyncio
import asyncio

# Define variables required for Session setup and Agent execution
APP_NAME="Google Search_agent"
USER_ID="user1234"
SESSION_ID="1234" 

# Define Agent with access to search tool 
root_agent = ADKAgent(
    name="basic_search_agent",
    model="gemini-2.0-flash-exp",
    description="Agent to answer questions using Google Search.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    tools=[google_search] # Google Search is a pre-built tool to perform Google searches. 
)

# Agent Interaction
async def call_agent(query):
    """
    Helper function to call the agent with a query.
    """
    
    # Session and Runner
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner( agent=root_agent, app_name=APP_NAME,
        session_service=session_service)
        content = types.Content(role='user', parts=[types.Part(text=query)]
        )

    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    
    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response) 

nest_asyncio.apply() 

asyncio.run(call_agent("what's the latest ai news?"))

```

## Code Example (Google Search)

This code demonstrates how to create and use a basic agent powered by the Google ADK for Python. The agent is designed to answer questions by utilizing Google Search as a tool.
```python
import os, getpass
import asyncio
import nest_asyncio from typing
import List from dotenv
import load_dotenv
import logging from google.adk.agents
import Agent as ADKAgent, LlmAgent from google.adk.runners
import Runner from google.adk.sessions
import InMemorySessionService from google.adk.tools
import google_search from google.adk.code_executors
import BuiltInCodeExecutor from google.genai 
import types 

# Define variables required for Session setup and Agent execution APP_NAME="calculator" USER_ID="user1234" SESSION_ID="session_code_exec_async" # Agent Definition code_agent = LlmAgent(   name="calculator_agent",   model="gemini-2.0-flash",   code_executor=BuiltInCodeExecutor(),   instruction="""You are a calculator agent.   When given a mathematical expression, write and execute Python code to calculate the result.   Return only the final numerical result as plain text, without markdown or code blocks.   """,   description="Executes Python code to perform calculations.", ) # Agent Interaction (Async) async def call_agent_async(query):   # Session and Runner   session_service = InMemorySessionService()   session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)   runner = Runner(agent=code_agent, app_name=APP_NAME, session_service=session_service)   content = types.Content(role='user', parts=[types.Part(text=query)])   print(f"\n--- Running Query: {query} ---")   final_response_text = "No final text response captured."   try:       # Use run_async       async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):           print(f"Event ID: {event.id}, Author: {event.author}")           # --- Check for specific parts FIRST ---           # has_specific_part = False           if event.content and event.content.parts and event.is_final_response():               for part in event.content.parts: # Iterate through all parts                   if part.executable_code:                       # Access the actual code string via .code                       print(f"  Debug: Agent generated code:\n```python\n{part.executable_code.code}\n```")                       has_specific_part = True                   elif part.code_execution_result:                       # Access outcome and output correctly                       print(f"  Debug: Code Execution Result: {part.code_execution_result.outcome} - Output:\n{part.code_execution_result.output}")                       has_specific_part = True                   # Also print any text parts found in any event for debugging                   elif part.text and not part.text.isspace():                       print(f"  Text: '{part.text.strip()}'")                       # Do not set has_specific_part=True here, as we want the final response logic below               # --- Check for final response AFTER specific parts ---               text_parts = [part.text for part in event.content.parts if part.text]               final_result = "".join(text_parts)               print(f"==> Final Agent Response: {final_result}")   except Exception as e:       print(f"ERROR during agent run: {e}")   print("-" * 30)

# Main async function to run the examples
async def main():
    await call_agent_async("Calculate the value of (5 + 7) * 3")
    await call_agent_async("What is 10 factorial?")

# Execute the main async function
try:
    nest_asyncio.apply()
    asyncio.run(main())
except RuntimeError as e:
    # Handle specific error when running asyncio.run in an already running loop (like Jupyter/Colab)
    if "cannot be called from a running event loop" in str(e):
        print("\nRunning in an existing event loop (like Colab/Jupyter).")
        print("Please run `await main()` in a notebook cell instead.")
        # If in an interactive environment like a notebook, you might need to run:
        # await main()
    else:
        raise e # Re-raise other runtime errors

```

## Code Example (Enterprise Search)

This code defines a Google ADK application using the google.adk library in Python. It specifically uses a VSearchAgent which is designed to answer questions by searching a specified Vertex AI Search datastore.

```python
import asyncio
from google.genai import types
from google.adk import agents
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import os

# --- Configuration --- # Ensure you have set your GOOGLE_API_KEY and DATASTORE_ID environment variables # For example: # os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" # os.environ["DATASTORE_ID"] = "YOUR_DATASTORE_ID" DATASTORE_ID = os.environ.get("DATASTORE_ID") # --- Application Constants --- APP_NAME = "vsearch_app" USER_ID = "user_123"  # Example User ID SESSION_ID = "session_456" # Example Session ID # --- Agent Definition (Updated with the newer model from the guide) --- vsearch_agent = agents.VSearchAgent(    name="q2_strategy_vsearch_agent",    description="Answers questions about Q2 strategy documents using Vertex AI Search.",    model="gemini-2.0-flash-exp", # Updated model based on the guide's examples    datastore_id=DATASTORE_ID,    model_parameters={"temperature": 0.0} ) # --- Runner and Session Initialization --- runner = Runner(    agent=vsearch_agent,    app_name=APP_NAME,    session_service=InMemorySessionService(), ) # --- Agent Invocation Logic --- async def call_vsearch_agent_async(query: str):    """Initializes a session and streams the agent's response."""    print(f"User: {query}")    print("Agent: ", end="", flush=True)    try:        # Construct the message content correctly        content = types.Content(role='user', parts=[types.Part(text=query)])        # Process events as they arrive from the asynchronous runner        async for event in runner.run_async(            user_id=USER_ID,            session_id=SESSION_ID,            new_message=content        ):            # For token-by-token streaming of the response text            if hasattr(event, 'content_part_delta') and event.content_part_delta:                print(event.content_part_delta.text, end="", flush=True)            # Process the final response and its associated metadata            if event.is_final_response():                print() # Newline after the streaming response                if event.grounding_metadata:                    print(f"  (Source Attributions: {len(event.grounding_metadata.grounding_attributions)} sources found)")                else:                    print("  (No grounding metadata found)")                print("-" * 30)    except Exception as e:        print(f"\nAn error occurred: {e}")        print("Please ensure your datastore ID is correct and that the service account has the necessary permissions.")        print("-" * 30) # --- Run Example --- async def run_vsearch_example():    # Replace with a question relevant to YOUR datastore content    await call_vsearch_agent_async("Summarize the main points about the Q2 strategy document.")    await call_vsearch_agent_async("What safety procedures are mentioned for lab X?") 

# --- Execution ---
if __name__ == "__main__":
    if not DATASTORE_ID:
        print("Error: DATASTORE_ID environment variable is not set.")
    else:
        try:
            asyncio.run(run_vsearch_example())
        except RuntimeError as e:
            # This handles cases where asyncio.run is called in an environment
            # that already has a running event loop (like a Jupyter notebook).
            if "cannot be called from a running event loop" in str(e):
                print("Skipping execution in a running event loop. Please run this script directly.")
            else:
                raise e

```
Overall, this code provides a basic framework for building a conversational AI application that leverages Vertex AI Search to answer questions based on information stored in a datastore. It demonstrates how to define an agent, set up a runner, and interact with the agent asynchronously while streaming the response.

