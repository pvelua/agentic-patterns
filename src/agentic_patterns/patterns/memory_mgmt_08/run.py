"""Memory Management pattern - Three examples demonstrating different memory types.

This module demonstrates:
1. Personal AI Assistant: ConversationBufferMemory + Semantic Memory
2. Customer Support Agent: ChatMessageHistory + Episodic + Procedural Memory
3. Adaptive Learning Tutor: All memory types combined
"""

import warnings
# Suppress Pydantic v1 deprecation warning for Python 3.14+
warnings.filterwarnings('ignore', message='.*Pydantic V1.*', category=UserWarning)

import sys
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory as InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.store.memory import InMemoryStore

from agentic_patterns.common.model_factory import ModelFactory
from agentic_patterns.common.output_writer import create_writer
from agentic_patterns.patterns.memory_mgmt_08.config import MemoryManagementConfig


# ============================================================================
# Example 1: Personal AI Assistant
# Demonstrates: ConversationBufferMemory (automated short-term)
#              + Semantic Memory (long-term facts)
# ============================================================================

def run_personal_assistant(
    model_name: str = "gpt-4o",
    conversation: List[str] = None,
    semantic_memory: Dict[str, str] = None,
    config: MemoryManagementConfig = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run personal AI assistant with conversation buffer and semantic memory.

    Args:
        model_name: LLM model to use
        conversation: List of user messages to process sequentially
        semantic_memory: Dictionary of user facts/preferences (semantic memory)
        config: Configuration object (uses default if None)
        verbose: Whether to print intermediate outputs

    Returns:
        Dictionary containing conversation history and semantic memory used
    """
    if config is None:
        config = MemoryManagementConfig()

    if semantic_memory is None:
        semantic_memory = config.default_semantic_memory

    if conversation is None:
        conversation = [
            "What's my preferred meeting time?",
            "Can you schedule a meeting with John for next Tuesday at that time?",
            "What restaurant should I book for lunch that day?",
            "Great, and what are my dietary restrictions again?"
        ]

    if verbose:
        print("=" * 80)
        print("EXAMPLE 1: Personal AI Assistant")
        print("Memory Types: ConversationBufferMemory + Semantic Memory")
        print("=" * 80)
        print("\n📋 SEMANTIC MEMORY (User Facts):")
        for key, value in list(semantic_memory.items())[:5]:  # Show first 5
            print(f"  • {key}: {value}")
        print(f"  ... and {len(semantic_memory) - 5} more facts\n")

    # Initialize LLM
    llm = ModelFactory.create(model_name, **config.get_assistant_kwargs())

    # Create ConversationBufferMemory (automated short-term memory)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )

    # Format semantic memory for prompt injection
    semantic_facts = "\n".join([f"- {key}: {value}" for key, value in semantic_memory.items()])

    # Create prompt template with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", config.assistant_system_prompt + f"\n\nUSER PROFILE (Semantic Memory):\n{semantic_facts}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create chain with memory
    chain = prompt | llm | StrOutputParser()

    # Process conversation
    conversation_history = []
    for user_message in conversation:
        if verbose:
            print(f"\n👤 User: {user_message}")

        # Get chat history from memory
        chat_history = memory.load_memory_variables({}).get("chat_history", [])

        # Run chain
        response = chain.invoke({
            "input": user_message,
            "chat_history": chat_history
        })

        if verbose:
            print(f"🤖 Assistant: {response}")

        # Save to memory (automated memory management)
        memory.save_context({"input": user_message}, {"output": response})

        conversation_history.append({
            "user": user_message,
            "assistant": response
        })

    if verbose:
        print(f"\n{'─' * 80}")
        print("📊 MEMORY STATUS:")
        print(f"  • Short-term (buffer): {len(memory.chat_memory.messages)} messages")
        print(f"  • Semantic facts: {len(semantic_memory)} stored")
        print("=" * 80)

    return {
        "conversation_history": conversation_history,
        "semantic_memory": semantic_memory,
        "total_messages": len(memory.chat_memory.messages)
    }


# ============================================================================
# Example 2: Customer Support Agent
# Demonstrates: ChatMessageHistory (manual short-term)
#              + Episodic Memory (past experiences)
#              + Procedural Memory (rules/procedures)
# ============================================================================

def run_customer_support(
    model_name: str = "gpt-4o",
    customer_query: str = None,
    episodic_memory: List[Dict[str, str]] = None,
    procedural_memory: List[Dict[str, str]] = None,
    config: MemoryManagementConfig = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run customer support agent with episodic and procedural memory.

    Args:
        model_name: LLM model to use
        customer_query: Customer's support request
        episodic_memory: List of past support tickets (episodic memory)
        procedural_memory: List of company policies/procedures (procedural memory)
        config: Configuration object (uses default if None)
        verbose: Whether to print intermediate outputs

    Returns:
        Dictionary containing support response and memory context used
    """
    if config is None:
        config = MemoryManagementConfig()

    if episodic_memory is None:
        episodic_memory = config.default_episodic_memory

    if procedural_memory is None:
        procedural_memory = config.default_procedural_memory

    if customer_query is None:
        customer_query = """I bought a SmartHome Hub from you 3 months ago and now the WiFi keeps
disconnecting every few hours. I've tried restarting it but the problem comes back.
This is really frustrating - I need this to work reliably. Can you help?"""

    if verbose:
        print("=" * 80)
        print("EXAMPLE 2: Customer Support Agent")
        print("Memory Types: ChatMessageHistory + Episodic + Procedural Memory")
        print("=" * 80)
        print(f"\n📞 CUSTOMER QUERY:\n{customer_query}\n")

    # Initialize LLM
    llm = ModelFactory.create(model_name, **config.get_support_kwargs())

    # Create ChatMessageHistory (manual memory management)
    chat_history = InMemoryChatMessageHistory()

    # Search episodic memory for relevant past tickets
    relevant_episodes = [ep for ep in episodic_memory
                        if "wifi" in ep["issue"].lower() or "smarthome" in ep["issue"].lower()]

    if verbose:
        print(f"🔍 EPISODIC MEMORY SEARCH:")
        print(f"  Found {len(relevant_episodes)} relevant past tickets:")
        for ep in relevant_episodes:
            print(f"  • [{ep['ticket_id']}] {ep['issue']}")
            print(f"    Resolution: {ep['resolution']}")
        print()

    # Search procedural memory for relevant policies/procedures
    relevant_procedures = [proc for proc in procedural_memory
                          if "wifi" in proc["rule"].lower() or "troubleshoot" in proc["rule"].lower()]

    if verbose:
        print(f"📖 PROCEDURAL MEMORY SEARCH:")
        print(f"  Found {len(relevant_procedures)} relevant procedures:")
        for proc in relevant_procedures:
            print(f"  • [{proc['rule_id']}] {proc['category']}: {proc['rule']}")
        print()

    # Build context from episodic memory
    episodic_context = "\n".join([
        f"Ticket {ep['ticket_id']} ({ep['date']}): {ep['issue']}\n"
        f"  → Solution: {ep['resolution']}\n"
        f"  → Outcome: {ep['outcome']}"
        for ep in relevant_episodes
    ])

    # Build context from procedural memory
    procedural_context = "\n".join([
        f"[{proc['rule_id']}] {proc['category'].upper()}: {proc['rule']}"
        for proc in relevant_procedures
    ])

    # Construct system message with long-term memory
    system_message = SystemMessage(content=f"""{config.support_system_prompt}

RELEVANT PAST TICKETS (Episodic Memory):
{episodic_context if episodic_context else "No directly relevant past tickets found."}

APPLICABLE PROCEDURES (Procedural Memory):
{procedural_context if procedural_context else "No specific procedures found."}

Use this information to provide informed, policy-compliant support.""")

    # Manually add to chat history
    chat_history.add_message(system_message)
    chat_history.add_user_message(customer_query)

    # Get response
    response = llm.invoke(chat_history.messages)

    # Manually add response to history
    chat_history.add_message(response)

    if verbose:
        print("💬 AGENT RESPONSE:")
        print(response.content)
        print(f"\n{'─' * 80}")
        print("📊 MEMORY STATUS:")
        print(f"  • Short-term (chat history): {len(chat_history.messages)} messages")
        print(f"  • Episodic memories used: {len(relevant_episodes)}")
        print(f"  • Procedural rules applied: {len(relevant_procedures)}")
        print("=" * 80)

    # Simulate follow-up question to demonstrate chat history
    follow_up = "How long will this take to fix?"
    if verbose:
        print(f"\n📞 FOLLOW-UP QUERY: {follow_up}")

    chat_history.add_user_message(follow_up)
    follow_up_response = llm.invoke(chat_history.messages)
    chat_history.add_message(follow_up_response)

    if verbose:
        print(f"💬 AGENT RESPONSE:\n{follow_up_response.content}")
        print(f"\n📊 Updated chat history: {len(chat_history.messages)} messages")
        print("=" * 80)

    return {
        "initial_query": customer_query,
        "initial_response": response.content,
        "follow_up_query": follow_up,
        "follow_up_response": follow_up_response.content,
        "episodic_memories_used": len(relevant_episodes),
        "procedural_rules_applied": len(relevant_procedures),
        "total_messages": len(chat_history.messages)
    }


# ============================================================================
# Example 3: Adaptive Learning Tutor
# Demonstrates: ALL memory types working together
#              - ConversationBufferMemory (short-term)
#              - Semantic Memory (subject knowledge)
#              - Episodic Memory (learning history)
#              - Procedural Memory (teaching strategies)
# ============================================================================

def run_learning_tutor(
    model_name: str = "gpt-4o",
    student_question: str = None,
    semantic_memory: Dict[str, str] = None,
    episodic_memory: List[Dict[str, str]] = None,
    procedural_memory: List[Dict[str, str]] = None,
    config: MemoryManagementConfig = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run adaptive learning tutor with all memory types.

    Args:
        model_name: LLM model to use
        student_question: Student's question or problem
        semantic_memory: Mathematical concepts knowledge base
        episodic_memory: Student's learning history
        procedural_memory: Teaching strategies and rules
        config: Configuration object (uses default if None)
        verbose: Whether to print intermediate outputs

    Returns:
        Dictionary containing tutoring response and all memory contexts used
    """
    if config is None:
        config = MemoryManagementConfig()

    if semantic_memory is None:
        semantic_memory = config.default_tutor_semantic_memory

    if episodic_memory is None:
        episodic_memory = config.default_tutor_episodic_memory

    if procedural_memory is None:
        procedural_memory = config.default_tutor_procedural_memory

    if student_question is None:
        student_question = "I need help with this problem: Calculate (3 + 2) × 4 - 6 ÷ 2"

    if verbose:
        print("=" * 80)
        print("EXAMPLE 3: Adaptive Learning Tutor")
        print("Memory Types: ConversationBuffer + Semantic + Episodic + Procedural")
        print("=" * 80)
        print(f"\n🎓 STUDENT QUESTION:\n{student_question}\n")

    # Initialize LLM
    llm = ModelFactory.create(model_name, **config.get_tutor_kwargs())

    # Create ConversationBufferMemory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )

    # Analyze student question to retrieve relevant memories

    # 1. Semantic Memory: Find relevant concepts
    relevant_concepts = {}
    question_lower = student_question.lower()
    if "order" in question_lower or "pemdas" in question_lower or any(op in student_question for op in ['+', '-', '×', '÷', '*', '/']):
        relevant_concepts["order_of_operations"] = semantic_memory["order_of_operations"]
    if "fraction" in question_lower or "/" in student_question:
        relevant_concepts["fractions"] = semantic_memory.get("fractions", "")

    # If no specific match, use order of operations (for the example)
    if not relevant_concepts:
        relevant_concepts["order_of_operations"] = semantic_memory["order_of_operations"]

    # 2. Episodic Memory: Find relevant past sessions
    relevant_episodes = []
    for ep in episodic_memory:
        if "order" in ep["topic"].lower():
            relevant_episodes.append(ep)

    # 3. Procedural Memory: Select teaching strategies
    relevant_strategies = []
    # Check student's past performance to select strategy
    if relevant_episodes:
        avg_mastery = sum(int(ep["mastery_level"].rstrip("%")) for ep in relevant_episodes) / len(relevant_episodes)
        if avg_mastery < 60:
            # Find scaffolding strategies for struggling students
            relevant_strategies = [p for p in procedural_memory if "scaffold" in p["category"] or "struggle" in p["rule"].lower()]
        elif avg_mastery > 85:
            # Find challenge strategies for excelling students
            relevant_strategies = [p for p in procedural_memory if "excel" in p["rule"].lower() or "challeng" in p["rule"].lower()]
        else:
            # Standard teaching strategies
            relevant_strategies = [p for p in procedural_memory if "feedback" in p["category"] or "review" in p["category"]]
    else:
        # Default strategies
        relevant_strategies = [p for p in procedural_memory if p["rule_id"] in ["TEACH-004", "TEACH-006"]]

    if verbose:
        print("🧠 MEMORY RETRIEVAL:")
        print(f"\n📚 Semantic Memory (Concepts):")
        for concept, knowledge in relevant_concepts.items():
            print(f"  • {concept}: {knowledge}")

        print(f"\n📖 Episodic Memory (Learning History):")
        if relevant_episodes:
            for ep in relevant_episodes:
                print(f"  • Session {ep['session_id']} ({ep['date']}): {ep['topic']}")
                print(f"    Performance: {ep['performance']}")
                print(f"    Mastery: {ep['mastery_level']}")
        else:
            print("  • No directly relevant past sessions found")

        print(f"\n🎯 Procedural Memory (Teaching Strategies):")
        for strategy in relevant_strategies:
            print(f"  • [{strategy['rule_id']}] {strategy['rule']}")
        print()

    # Build comprehensive context
    semantic_context = "\n".join([f"- {concept}: {knowledge}" for concept, knowledge in relevant_concepts.items()])

    episodic_context = "\n".join([
        f"Session {ep['session_id']} ({ep['date']}): {ep['topic']}\n"
        f"  Performance: {ep['performance']}\n"
        f"  Mistakes: {ep['mistakes']}\n"
        f"  Mastery: {ep['mastery_level']}"
        for ep in relevant_episodes
    ]) if relevant_episodes else "This appears to be a new topic for the student."

    procedural_context = "\n".join([
        f"[{strategy['rule_id']}] {strategy['rule']}"
        for strategy in relevant_strategies
    ])

    # Create enhanced system prompt with all memory types
    enhanced_system_prompt = f"""{config.tutor_system_prompt}

RELEVANT CONCEPTS (Semantic Memory):
{semantic_context}

STUDENT'S LEARNING HISTORY (Episodic Memory):
{episodic_context}

TEACHING STRATEGIES TO APPLY (Procedural Memory):
{procedural_context}

Based on the student's history and the teaching strategies, adapt your response appropriately."""

    # Create prompt with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", enhanced_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create chain
    chain = prompt | llm | StrOutputParser()

    # Get response
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    response = chain.invoke({
        "input": student_question,
        "chat_history": chat_history
    })

    # Save to memory
    memory.save_context({"input": student_question}, {"output": response})

    if verbose:
        print("👨‍🏫 TUTOR RESPONSE:")
        print(response)
        print()

    # Simulate follow-up to show short-term memory in action
    follow_up = "Can you show me one more example like that?"
    if verbose:
        print(f"🎓 STUDENT FOLLOW-UP: {follow_up}")

    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    follow_up_response = chain.invoke({
        "input": follow_up,
        "chat_history": chat_history
    })

    memory.save_context({"input": follow_up}, {"output": follow_up_response})

    if verbose:
        print(f"👨‍🏫 TUTOR RESPONSE:\n{follow_up_response}")
        print(f"\n{'─' * 80}")
        print("📊 MEMORY STATUS:")
        print(f"  • Short-term (buffer): {len(memory.chat_memory.messages)} messages")
        print(f"  • Semantic concepts used: {len(relevant_concepts)}")
        print(f"  • Episodic sessions referenced: {len(relevant_episodes)}")
        print(f"  • Procedural strategies applied: {len(relevant_strategies)}")
        print("=" * 80)

    return {
        "initial_question": student_question,
        "initial_response": response,
        "follow_up_question": follow_up,
        "follow_up_response": follow_up_response,
        "semantic_concepts_used": len(relevant_concepts),
        "episodic_sessions_referenced": len(relevant_episodes),
        "procedural_strategies_applied": len(relevant_strategies),
        "total_messages": len(memory.chat_memory.messages)
    }


# ============================================================================
# Example 4: Personal Finance Advisor (InMemoryStore - Production Pattern)
# Demonstrates: ConversationBufferMemory (short-term)
#              + InMemoryStore for Semantic + Episodic + Procedural (long-term)
# ============================================================================

async def run_finance_advisor(
    model_name: str = "gpt-4o",
    client_question: str = None,
    semantic_memory: Dict[str, str] = None,
    episodic_memory: List[Dict[str, str]] = None,
    procedural_memory: List[Dict[str, str]] = None,
    config: MemoryManagementConfig = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run personal finance advisor with InMemoryStore for production-like memory management.

    This example demonstrates LangGraph's InMemoryStore for managing long-term memories
    as JSON documents with proper namespacing and async operations.

    Args:
        model_name: LLM model to use
        client_question: Client's financial question
        semantic_memory: Financial knowledge base (will be stored in InMemoryStore)
        episodic_memory: Client's financial history (will be stored in InMemoryStore)
        procedural_memory: Financial strategies and protocols (will be stored in InMemoryStore)
        config: Configuration object (uses default if None)
        verbose: Whether to print intermediate outputs

    Returns:
        Dictionary containing advisor response and memory statistics
    """
    if config is None:
        config = MemoryManagementConfig()

    if semantic_memory is None:
        semantic_memory = config.default_advisor_semantic_memory

    if episodic_memory is None:
        episodic_memory = config.default_advisor_episodic_memory

    if procedural_memory is None:
        procedural_memory = config.default_advisor_procedural_memory

    if client_question is None:
        client_question = """I'm 35 years old and currently contributing 15% to my 401(k).
I just got a raise and have an extra $500/month. Should I put it all into my 401(k),
or should I open a Roth IRA? I'm not sure what the tax implications are."""

    if verbose:
        print("=" * 80)
        print("EXAMPLE 4: Personal Finance Advisor")
        print("Memory Types: ConversationBuffer + InMemoryStore (Semantic + Episodic + Procedural)")
        print("=" * 80)
        print(f"\n💰 CLIENT QUESTION:\n{client_question}\n")

    # Initialize LLM
    llm = ModelFactory.create(model_name, **config.get_advisor_kwargs())

    # Create InMemoryStore instances for long-term memory
    # Each store uses namespaces to organize different types of memories
    memory_store = InMemoryStore()

    if verbose:
        print("🔧 INITIALIZING INMEMORYSTORE...")
        print("  Creating separate namespaces for semantic, episodic, and procedural memories\n")

    # Populate InMemoryStore with semantic memory (financial knowledge)
    for key, value in semantic_memory.items():
        await memory_store.aput(
            namespace=("advisor", "semantic"),
            key=key,
            value={"concept": key, "knowledge": value}
        )

    # Populate InMemoryStore with episodic memory (client's financial history)
    for episode in episodic_memory:
        await memory_store.aput(
            namespace=("advisor", "episodic"),
            key=episode["session_id"],
            value=episode
        )

    # Populate InMemoryStore with procedural memory (strategies and protocols)
    for procedure in procedural_memory:
        await memory_store.aput(
            namespace=("advisor", "procedural"),
            key=procedure["rule_id"],
            value=procedure
        )

    if verbose:
        print("✅ INMEMORYSTORE POPULATED:")
        print(f"  • Semantic memories: {len(semantic_memory)} concepts stored")
        print(f"  • Episodic memories: {len(episodic_memory)} sessions stored")
        print(f"  • Procedural memories: {len(procedural_memory)} rules stored\n")

    # Search InMemoryStore for relevant memories
    if verbose:
        print("🔍 SEARCHING INMEMORYSTORE FOR RELEVANT MEMORIES:\n")

    # Search semantic memory for relevant concepts
    question_lower = client_question.lower()
    relevant_semantic = []

    # Search for tax-related concepts
    if "tax" in question_lower or "401" in question_lower or "roth" in question_lower:
        if "401" in question_lower:
            tax_401k = await memory_store.aget(namespace=("advisor", "semantic"), key="tax_401k")
            if tax_401k:
                relevant_semantic.append(tax_401k.value)
        if "roth" in question_lower:
            roth_ira = await memory_store.aget(namespace=("advisor", "semantic"), key="tax_roth_ira")
            if roth_ira:
                relevant_semantic.append(roth_ira.value)

    # Search episodic memory for relevant past sessions
    # Use asearch to get all items from episodic namespace (namespace_prefix is positional-only)
    episodic_items = await memory_store.asearch(
        ("advisor", "episodic"),  # namespace_prefix (positional)
        limit=100  # High limit to get all items
    )

    relevant_episodic = []
    for item in episodic_items:
        episode = item.value
        if "401" in episode.get("topic", "").lower() or "retirement" in episode.get("topic", "").lower():
            relevant_episodic.append(episode)

    # Search procedural memory for relevant strategies
    procedural_items = await memory_store.asearch(
        ("advisor", "procedural"),  # namespace_prefix (positional)
        limit=100  # High limit to get all items
    )

    relevant_procedural = []
    for item in procedural_items:
        procedure = item.value
        if "investment" in procedure.get("category", "").lower() or "retirement" in procedure.get("rule", "").lower():
            relevant_procedural.append(procedure)

    if verbose:
        print("📚 Semantic Memory Retrieved:")
        for sem in relevant_semantic:
            print(f"  • {sem['concept']}: {sem['knowledge'][:80]}...")

        print(f"\n📖 Episodic Memory Retrieved:")
        for ep in relevant_episodic:
            print(f"  • Session {ep['session_id']} ({ep['date']}): {ep['topic']}")
            print(f"    Decision: {ep['decision']}")

        print(f"\n🎯 Procedural Memory Retrieved:")
        for proc in relevant_procedural:
            print(f"  • [{proc['rule_id']}] {proc['rule']}")
        print()

    # Build context from InMemoryStore retrievals
    semantic_context = "\n".join([
        f"- {mem['concept']}: {mem['knowledge']}"
        for mem in relevant_semantic
    ]) if relevant_semantic else "No directly relevant financial concepts found in knowledge base."

    episodic_context = "\n".join([
        f"Session {ep['session_id']} ({ep['date']}): {ep['topic']}\n"
        f"  Decision: {ep['decision']}\n"
        f"  Context: {ep['context']}\n"
        f"  Outcome: {ep['outcome']}"
        for ep in relevant_episodic
    ]) if relevant_episodic else "No directly relevant past sessions found."

    procedural_context = "\n".join([
        f"[{proc['rule_id']}] {proc['category'].upper()}: {proc['rule']}"
        for proc in relevant_procedural
    ]) if relevant_procedural else "No specific strategies found."

    # Create enhanced system prompt with InMemoryStore data
    enhanced_system_prompt = f"""{config.advisor_system_prompt}

RELEVANT FINANCIAL KNOWLEDGE (from Semantic Memory Store):
{semantic_context}

CLIENT'S FINANCIAL HISTORY (from Episodic Memory Store):
{episodic_context}

APPLICABLE STRATEGIES & PROTOCOLS (from Procedural Memory Store):
{procedural_context}

Based on the client's history and applicable strategies, provide personalized, consistent advice."""

    # Create ConversationBufferMemory for short-term conversation context
    conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )

    # Create prompt with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", enhanced_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create chain
    chain = prompt | llm | StrOutputParser()

    # Get response
    chat_history = conversation_memory.load_memory_variables({}).get("chat_history", [])
    response = chain.invoke({
        "input": client_question,
        "chat_history": chat_history
    })

    # Save to conversation memory
    conversation_memory.save_context({"input": client_question}, {"output": response})

    if verbose:
        print("💼 ADVISOR RESPONSE:")
        print(response)
        print()

    # Simulate follow-up to demonstrate short-term memory
    follow_up = "That makes sense. Can you remind me what my current 401(k) contribution rate is?"
    if verbose:
        print(f"💰 CLIENT FOLLOW-UP: {follow_up}")

    chat_history = conversation_memory.load_memory_variables({}).get("chat_history", [])
    follow_up_response = chain.invoke({
        "input": follow_up,
        "chat_history": chat_history
    })

    conversation_memory.save_context({"input": follow_up}, {"output": follow_up_response})

    if verbose:
        print(f"💼 ADVISOR RESPONSE:\n{follow_up_response}")
        print(f"\n{'─' * 80}")
        print("📊 MEMORY STATUS:")
        print(f"  • Short-term (conversation buffer): {len(conversation_memory.chat_memory.messages)} messages")
        print(f"  • Semantic memories retrieved from InMemoryStore: {len(relevant_semantic)}")
        print(f"  • Episodic memories retrieved from InMemoryStore: {len(relevant_episodic)}")
        print(f"  • Procedural strategies retrieved from InMemoryStore: {len(relevant_procedural)}")
        print(f"\n  💾 INMEMORYSTORE STATISTICS:")

        # Get stats from InMemoryStore using asearch
        all_semantic = await memory_store.asearch(
            ("advisor", "semantic"),  # namespace_prefix (positional)
            limit=100
        )

        all_episodic_items = await memory_store.asearch(
            ("advisor", "episodic"),  # namespace_prefix (positional)
            limit=100
        )

        all_procedural_items = await memory_store.asearch(
            ("advisor", "procedural"),  # namespace_prefix (positional)
            limit=100
        )

        print(f"     Total semantic concepts in store: {len(all_semantic)}")
        print(f"     Total episodic sessions in store: {len(all_episodic_items)}")
        print(f"     Total procedural rules in store: {len(all_procedural_items)}")
        print("=" * 80)

    return {
        "initial_question": client_question,
        "initial_response": response,
        "follow_up_question": follow_up,
        "follow_up_response": follow_up_response,
        "semantic_memories_retrieved": len(relevant_semantic),
        "episodic_memories_retrieved": len(relevant_episodic),
        "procedural_strategies_retrieved": len(relevant_procedural),
        "total_conversation_messages": len(conversation_memory.chat_memory.messages),
        "total_semantic_in_store": len(semantic_memory),
        "total_episodic_in_store": len(episodic_memory),
        "total_procedural_in_store": len(procedural_memory)
    }


# ============================================================================
# Model Comparison
# ============================================================================

def compare_models(
    models: List[str] = None,
    example: str = "assistant",
    config: MemoryManagementConfig = None
) -> Dict[str, Any]:
    """
    Compare different models on the same memory management task.

    Args:
        models: List of model names to compare
        example: Which example to run ("assistant", "support", "tutor", or "advisor")
        config: Configuration object (uses default if None)

    Returns:
        Dictionary mapping model names to their results
    """
    if models is None:
        models = ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-5-20250929"]

    if config is None:
        config = MemoryManagementConfig()

    writer = create_writer("08_memory_mgmt")
    results = {}

    print(f"\n{'=' * 80}")
    print(f"COMPARING MODELS ON: {example.upper()} EXAMPLE")
    print(f"Models: {', '.join(models)}")
    print(f"{'=' * 80}\n")

    for model in models:
        print(f"\n{'─' * 80}")
        print(f"Testing model: {model}")
        print(f"{'─' * 80}")

        try:
            if example == "assistant":
                result = run_personal_assistant(model_name=model, config=config, verbose=True)
            elif example == "support":
                result = run_customer_support(model_name=model, config=config, verbose=True)
            elif example == "tutor":
                result = run_learning_tutor(model_name=model, config=config, verbose=True)
            elif example == "advisor":
                # Run async function
                result = asyncio.run(run_finance_advisor(model_name=model, config=config, verbose=True))
            else:
                raise ValueError(f"Unknown example: {example}")

            results[model] = result

        except Exception as e:
            print(f"❌ Error with model {model}: {e}")
            results[model] = {"error": str(e)}

    # Write comparison results
    metadata = {
        "example": example,
        "num_models": len(models),
        "timestamp": datetime.now().isoformat()
    }

    writer.write_comparison(
        results=results,
        input_data={"example": example},
        metadata=metadata
    )

    return results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
    example_name = sys.argv[2] if len(sys.argv) > 2 else "all"

    config = MemoryManagementConfig()
    writer = create_writer("08_memory_mgmt")

    print(f"\n{'=' * 80}")
    print("MEMORY MANAGEMENT PATTERN EXAMPLES")
    print(f"Model: {model_name}")
    print(f"{'=' * 80}\n")

    results = {}

    if example_name in ["all", "assistant"]:
        print("\n" + "=" * 80)
        result = run_personal_assistant(model_name=model_name, config=config, verbose=True)
        results["personal_assistant"] = result

        writer.write_result(
            result=result,
            model_name=model_name,
            input_data={"example": "personal_assistant"},
            metadata={"workflow_type": "conversation_buffer_semantic"}
        )

    if example_name in ["all", "support"]:
        print("\n" + "=" * 80)
        result = run_customer_support(model_name=model_name, config=config, verbose=True)
        results["customer_support"] = result

        writer.write_result(
            result=result,
            model_name=model_name,
            input_data={"example": "customer_support"},
            metadata={"workflow_type": "chat_history_episodic_procedural"}
        )

    if example_name in ["all", "tutor"]:
        print("\n" + "=" * 80)
        result = run_learning_tutor(model_name=model_name, config=config, verbose=True)
        results["learning_tutor"] = result

        writer.write_result(
            result=result,
            model_name=model_name,
            input_data={"example": "learning_tutor"},
            metadata={"workflow_type": "all_memory_types"}
        )

    if example_name in ["advisor"]:
        print("\n" + "=" * 80)
        result = asyncio.run(run_finance_advisor(model_name=model_name, config=config, verbose=True))
        results["finance_advisor"] = result

        writer.write_result(
            result=result,
            model_name=model_name,
            input_data={"example": "finance_advisor"},
            metadata={"workflow_type": "inmemorystore_all_memory_types"}
        )

    print(f"\n{'=' * 80}")
    print("✅ All examples completed successfully!")
    print(f"Results saved to experiments/results/")
    print(f"{'=' * 80}\n")
