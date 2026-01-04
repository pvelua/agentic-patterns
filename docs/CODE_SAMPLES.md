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

The **Reflection pattern** involves an agent evaluating its own work, output, or internal state and using that evaluation to improve its performance or refine its response. It’s a form of self-correction or self-improvement, allowing the agent to iteratively refine its output or adjust its approach based on feedback, internal critique, or comparison against desired criteria. Reflection can occasionally be facilitated by a separate agent whose specific role is to analyze the output of an initial agent.


