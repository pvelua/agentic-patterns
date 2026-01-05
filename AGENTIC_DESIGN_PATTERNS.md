# Agentic Design Patterns

In simple terms, an AI agent is a system designed to perceive its environment and take actions to achieve a specific goal. It is an evolution from a standard large language model (LLM), enhanced with the abilities to plan, use tools, and interact with its surroundings. Itr follows simple five-step look to get things donbe:
1. **Get the mission:** You give it a goal, like “organize my schedule.” 
2. **Scan the scene:** It gathers all the necessary information—reading emails, checking calendars, and accessing contacts—to understand what is happening.
3. **Think it through:** It devises a plan of action by considering the optimal approach to achieve the goal. 
4. **Take action:** It executes the plan by sending invitations, scheduling meetings, and updating your calendar. 
5. **Learn and get better:** It observes successful outcomes and adapts accordingly. For example, if a meeting is rescheduled, the system learns from this event to enhance its future performance.

 Agent Complexity Levels:
 - **Level 0: The core reasoning engine** - the LLM operates without tools, memory, or environmental interaction, responding solely based on its pretrained knowledge.
 - **Level 1: The connected problme solver** - the LLM becomes a functional agent by connecting to and utilizing external tools. Ability to interact with the outside world across multiple steps is the core capability of a Level 1 agent.
 - **Level 2: The strategic problem solver** - the agent moves beyond single-tool use to tackle complex, multipart problems through strategic problem-solving. As it executes a sequence of actions, it actively performs context engineering: the strategic process of selecting, packaging, and managing the most relevant information for each step. To achieve maximum accuracy from an AI, it must be given a short, focused, and powerful context. Finally, the agent achieves self-improvement by refining iys own context engineering.
 - **The Level 3:The Collaborative Multi-Agent Systems** - a significant paradigm shift in AI development, moving away from the pursuit of a single, all-powerful superagent towards the rise of sophisticated, collaborative multi-agent systems.

The Future of Agents:
- **Hypothesis 1: The Emergence of the Generalist Agent** - first hypothesis is that AI agents will evolve from narrow specialists into true generalists capable of managing complex, ambiguous, and long-term goals with high reliability. An alternative, yet not mutually exclusive, approach is the rise of small language models (SLMs). This “Lego-like” concept involves composing systems from small, specialized expert agents rather than scaling up a single monolithic model.
- **Hypothesis 2: Deep Personalization and Proactive Goal Discovery** - second hypothesis posits that agents will become deeply personalized and proactive partners. AI systems operate as agents when they move beyond simply responding to chats or instructions.
- **Hypothesis 3: Embodiment and Physical World Interaction** - this hypothesis foresees agents breaking free from their purely digital confines to operate in the physical world. By integrating agentic AI with robotics, we will see the rise of “embodied agents.”
- **Hypothesis 4: The Agent-Driven Economy** - fourth hypothesis is that highly autonomous agents will become active participants in the economy, creating new markets and business models. We could see agents acting as independent economic entities, tasked with maximizing a specific outcome, such as profit.
- **Hypothesis 5: The Goal-Driven, Metamorphic Multi-Agent System** - this hypothesis posits the emergence of intelligent systems that operate not from explicit programming, but from a declared goal. The user simply states the desired outcome, and the system autonomously figures out how to achieve it. This marks a fundamental shift towards metamorphic multi-agent systems capable of true self-improvement at both the individual and collective levels. This system would be a dynamic entity, not a single agent. It would

## Pattern 1 - Prompt Chaining

The core idea is to break down the original, daunting problem into a sequence of smaller, more manageable sub-problems. Each sub-problem is addressed individually through a specifically designed prompt, and the output generated from one prompt is strategically fed as input into the subsequent prompt in the chain.

Furthermore, prompt chaining is not just about breaking down problems; it also enables the integration of external knowledge and tools. At each step, the LLM can be instructed to interact with external systems, APIs, or databases, enriching its knowledge and abilities beyond its internal training data. This capability dramatically expands the potential of LLMs, allowing them to function not just as isolated models but as integral components of broader, more intelligent systems.

### The Role of Structured Output
The reliability of a prompt chain is highly dependent on the integrity of the data passed between steps. If the output of one prompt is ambiguous or poorly formatted, the subsequent prompt may fail due to faulty input. To mitigate this, specifying a structured output format, such as JSON or XML, is crucial.
```json
{
    "trends": [
        {
            "trend_name": "AI-Powered Personalization",      
            "supporting_data": "73% of consumers prefer to do business with brands that use personal information to make their shopping experiences more relevant."
        },    
        {      
            "trend_name": "Sustainable and Ethical Brands",      
            "supporting_data": "Sales of products with ESG-related claims grew 28% over the last five years, compared to 20% for products without."    
        }  
    ] 
}
```
This structured format ensures that the data is machine-readable and can be precisely parsed and inserted into the next prompt without ambiguity.

### Practical Applications and Use Cases

1. **Infiormation processing workflow** - processing raw information through multiple transformations.
2. **Complex Query Answering** - answering complex questions that require multiple steps of reasoning or informationm retrieval.
3. **Data Extrcation and Transformation** - conversion of unstructured text into a structured format is typically achieved through an iterative process, requiring sequential modifications to improve the accuracy and completeness of the output.
4. **Content Generation Workflow** - compposition of complex content is a procedural task that is typically decomposed into distinct phases, including initial ideation, structural outlining, draftomg and subsequent revision.
5. **Conversational Agents with State** - comprehensive state management architectures employ methods more complex than sequential linking, prompt chaining provides a foundational mechanism for preserving conversational continuity.
6. **Code Generation and Refinemenmt** - a multi-stage process, requiring a problem to be decomposed into a sequence of discrete logical operations that are executed progressively.
7. **Multi-modal and Multi-step Reasoning** - analyzing datasets with diverse modalities necessitates breaking down the problem into smaller, prompt-based tasks.

### Context vs. Prompt Engineering

Context Engineering is the systematic discipline of designing, constructing, and delivering a complete informational environment to an AI model prior to token generation. This methodology asserts that the quality of a model’s output is less dependent on the model’s architecture itself and more on the richness of the context provided.

Prompt engineering is a subtask of context engineering along with using RAG & structured inputs / outputs, State, State, History and Memory management. The "engineering" invloves creating robust pipelines to fetch and transform required data at runtime and establishing feedback loopsto continually improve context quality.

## Pattern 2 - Routing

Routing introduces conditional logic into an agent’s operational framework, enabling a shift from a fixed execution path to a model where the agent dynamically evaluates specific criteria to select from a set of possible subsequent actions. This allows for more flexible and context-aware system behavior.

Routing mechanism:
- **LLM-based Routing** - LLM can be prompted to analyze the input and output a specific identifier or instruction that indicates the next step.
- **Embedding-based Routing** - input query can be converted into vector embedding and then compared to embeddings representing different routes.
- **Rule-based Routing** - routing based on pre-defined rules or logic
- **Machine Learning Model-base Routing** - routing based on the output generated by a classifier model that has been trained on a small corpus of labeled datra.

## Pattern 3 - Parallelism

## Pattern 4 - Reflection

- [Google Agent Developer Kit (ADK) Documentation (Multi-Agent Systems)](https://​google.​github.​io/​adk-docs/​agents/​multi-agents/​)
- [LangChain Expression Language (LCEL) Documentation](https://​python.​langchain.​com/​docs/​introduction/​)
- [LangGraph Documentation](https://​www.​langchain.​com/​langgraph)
- [Training Language Models to Self-Correct via Reinforcement Learning](https://​arxiv.​org/​abs/​2409.​12917)

## Pattern 5 - Tool Use (Function Calling)

The Tool Use pattern, often implemented through a mechanism called Function Calling, enables an agent to interact with external APIs, databases, services, or even execute code. It allows the LLM at the core of the agent to decide when and how to use a specific external function based on the user’s request or the current state of the task. The process typically involves: 
1. Tool Definition: External functions or capabilities are defined and described to the LLM.
2. LLM Decision: The LLM receives the user’s request and the available tool definitions. Based on its understanding of the request and the tools, the LLM decides if calling one or more tools is necessary to fulfill the request. 
3. Function Call Generation: If the LLM decides to use a tool, it generates a structured output (often a JSON object) that specifies the name of the tool to call and the arguments (parameters) to pass to it, extracted from the user’s request. 
4. Tool Execution: The agentic framework or orchestration layer intercepts this structured output. It identifies the requested tool and executes the actual external function with the provided arguments. 
5. Observation/Result: The output or result from the tool execution is returned to the agent. 
6. LLM Processing (Optional but common): The LLM receives the tool’s output as context and uses it to formulate a final response to the user or decide on the next step in the workflow.

 Typical Use Cases:
 - Information retrieval from extrnal source
 - Interacting with databases and APIs
 - Performing calculations and data analysis
 - Sending communications
 - Executing code
 - Controlling other systems or devices

- [CrewAI Documentation (Tools)](https://​docs.​crewai.​com/​concepts/​tools)
- [Google Agent Developer Kit (ADK) Documentation (Tools)](https://​google.​github.​io/​adk-docs/​tools/​)
- [LangChain Documentation (Tools)](https://​python.​langchain.​com/​docs/​integrations/​tools/​)
- [OpenAI Function Calling Documentation](https://​platform.​openai.​com/​docs/​guides/​function-calling)



 








 






