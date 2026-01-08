"""
Multi-Agent Collaboration Pattern - Three different collaboration models using LangGraph.

This pattern demonstrates:
1. Sequential Pipeline: Research Paper Analysis (Researcher -> Critic -> Summarizer)
2. Parallel + Synthesis: Product Launch Campaign (Marketing + Content + Analyst -> Coordinator)
3. Multi-Perspective Debate: Code Review System (Security + Performance + Quality -> Synthesizer)
"""

import warnings
# Suppress Pydantic v1 deprecation warning for Python 3.14+
warnings.filterwarnings('ignore', message='.*Pydantic V1.*', category=UserWarning)

from typing import TypedDict, Annotated
import operator
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from agentic_patterns.common import ModelFactory, create_writer
from agentic_patterns.patterns.multi_agent_collab_07.config import MultiAgentCollabConfig


# ============================================================================
# EXAMPLE 1: RESEARCH PAPER ANALYSIS (SEQUENTIAL PIPELINE)
# ============================================================================

class ResearchAnalysisState(TypedDict):
    """State for sequential research paper analysis workflow."""
    paper_content: str  # The input paper content
    research_analysis: str  # Output from researcher
    critical_review: str  # Output from critic
    final_summary: str  # Output from summarizer
    messages: Annotated[list, operator.add]  # Message history


def run_research_analysis(
    model_name: str = "gpt-4o",
    paper_content: str = None,
    config: MultiAgentCollabConfig = None,
    verbose: bool = True
):
    """
    Run the sequential research paper analysis workflow.

    Agents work in sequence:
    1. Researcher: Extracts key information from paper
    2. Critic: Evaluates methodology and validity
    3. Summarizer: Creates final comprehensive summary

    Args:
        model_name: Model to use for all agents
        paper_content: Research paper content to analyze
        config: Pattern configuration
        verbose: Whether to print intermediate results

    Returns:
        Dictionary containing all analysis stages and final summary
    """
    config = config or MultiAgentCollabConfig()

    # Default paper content if none provided
    if paper_content is None:
        paper_content = """Title: Efficient Attention Mechanisms for Large Language Models

Abstract: This paper presents a novel attention mechanism that reduces computational complexity
from O(n²) to O(n log n) while maintaining model performance. We introduce sparse attention
patterns based on learned importance scores, allowing models to focus on relevant context.

Methodology: We trained transformer models with our proposed attention mechanism on multiple
datasets including WikiText-103 and C4. Models ranged from 125M to 1.3B parameters. We compared
against standard self-attention and existing sparse attention methods.

Results: Our approach achieved 95% of full attention performance while reducing training time
by 40% and inference time by 60%. Memory usage decreased by 45% for long sequences (>2048 tokens).

Conclusions: Learned sparse attention patterns can significantly improve efficiency without
sacrificing model quality. This enables scaling to longer context windows and larger models."""

    # Initialize the Language Models
    researcher_llm = ModelFactory.create(model_name, **config.get_researcher_kwargs())
    critic_llm = ModelFactory.create(model_name, **config.get_critic_kwargs())
    summarizer_llm = ModelFactory.create(model_name, **config.get_synthesizer_kwargs())

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"EXAMPLE 1: Research Paper Analysis (Sequential Pipeline)")
        print(f"Model: {model_name}")
        print(f"{'=' * 80}\n")

    # --- Define Workflow Nodes ---

    def researcher_node(state: ResearchAnalysisState) -> ResearchAnalysisState:
        """Research agent extracts key information from the paper."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("AGENT 1: RESEARCHER (Extracting key information)")
            print(f"{'─' * 80}")

        messages = [
            SystemMessage(content=config.researcher_system_prompt),
            HumanMessage(content=f"Analyze this research paper:\n\n{state['paper_content']}")
        ]

        response = researcher_llm.invoke(messages)
        analysis = response.content

        if verbose:
            print(f"\n📊 Research Analysis:")
            print(analysis)

        return {
            **state,
            "research_analysis": analysis,
            "messages": messages + [response]
        }

    def critic_node(state: ResearchAnalysisState) -> ResearchAnalysisState:
        """Critic agent evaluates the research methodology and findings."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("AGENT 2: CRITIC (Evaluating methodology)")
            print(f"{'─' * 80}")

        messages = [
            SystemMessage(content=config.critic_system_prompt),
            HumanMessage(content=f"""Review this research analysis:

Original Paper:
{state['paper_content']}

Research Analysis:
{state['research_analysis']}

Provide a critical evaluation of the methodology, findings, and limitations.""")
        ]

        response = critic_llm.invoke(messages)
        critique = response.content

        if verbose:
            print(f"\n🔍 Critical Review:")
            print(critique)

        return {
            **state,
            "critical_review": critique,
            "messages": state["messages"] + messages + [response]
        }

    def summarizer_node(state: ResearchAnalysisState) -> ResearchAnalysisState:
        """Summarizer creates final comprehensive summary."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("AGENT 3: SUMMARIZER (Creating final summary)")
            print(f"{'─' * 80}")

        messages = [
            SystemMessage(content=config.summarizer_system_prompt),
            HumanMessage(content=f"""Create a comprehensive summary combining these inputs:

Research Analysis:
{state['research_analysis']}

Critical Review:
{state['critical_review']}

Synthesize this into a balanced, informative summary.""")
        ]

        response = summarizer_llm.invoke(messages)
        summary = response.content

        if verbose:
            print(f"\n📝 Final Summary:")
            print(summary)

        return {
            **state,
            "final_summary": summary,
            "messages": state["messages"] + messages + [response]
        }

    # --- Build the Workflow Graph ---
    workflow = StateGraph(ResearchAnalysisState)

    # Add nodes in sequence
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("summarizer", summarizer_node)

    # Define sequential flow
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "critic")
    workflow.add_edge("critic", "summarizer")
    workflow.add_edge("summarizer", END)

    # Compile and run
    app = workflow.compile()

    initial_state = {
        "paper_content": paper_content,
        "research_analysis": "",
        "critical_review": "",
        "final_summary": "",
        "messages": []
    }

    final_state = app.invoke(initial_state)

    result = {
        "paper_content": paper_content[:200] + "...",
        "research_analysis": final_state["research_analysis"],
        "critical_review": final_state["critical_review"],
        "final_summary": final_state["final_summary"],
        "workflow_type": "sequential_pipeline",
        "num_agents": 3
    }

    if verbose:
        print(f"\n{'═' * 80}")
        print("SEQUENTIAL WORKFLOW COMPLETE")
        print(f"{'═' * 80}\n")

    # Write results
    writer = create_writer("07_multi_agent_collab")
    log_path = writer.write_result(
        result=result,
        model_name=model_name,
        input_data={"paper_content": paper_content[:200] + "..."},
        metadata={
            "example": "research_analysis",
            "workflow_type": "sequential_pipeline",
            "num_agents": 3
        }
    )

    if verbose:
        print(f"Results written to: {log_path}\n")

    return result


# ============================================================================
# EXAMPLE 2: PRODUCT LAUNCH CAMPAIGN (PARALLEL + SYNTHESIS)
# ============================================================================

class ProductLaunchState(TypedDict):
    """State for parallel product launch campaign workflow."""
    product_description: str  # Input product description
    marketing_strategy: str  # Output from marketing strategist
    content_ideas: str  # Output from content writer
    market_analysis: str  # Output from data analyst
    launch_plan: str  # Final synthesized plan from coordinator
    messages: Annotated[list, operator.add]


def run_product_launch(
    model_name: str = "gpt-4o",
    product_description: str = None,
    config: MultiAgentCollabConfig = None,
    verbose: bool = True
):
    """
    Run the parallel product launch campaign workflow.

    Agents work in parallel, then synthesize:
    1. Marketing Strategist: Develops marketing strategy
    2. Content Writer: Creates content ideas
    3. Data Analyst: Provides market analysis
    4. Coordinator: Synthesizes all inputs into launch plan

    Args:
        model_name: Model to use for all agents
        product_description: Product to create launch campaign for
        config: Pattern configuration
        verbose: Whether to print intermediate results

    Returns:
        Dictionary containing all contributions and final launch plan
    """
    config = config or MultiAgentCollabConfig()

    if product_description is None:
        product_description = """Product: EcoTrack Smart Home Energy Monitor

Description: A WiFi-enabled device that tracks real-time energy consumption of individual
appliances and provides AI-powered recommendations to reduce energy costs and carbon footprint.

Key Features:
- Real-time monitoring of individual appliances
- AI-powered energy saving recommendations
- Mobile app with detailed analytics
- Integration with smart home systems
- Carbon footprint tracking
- Automatic cost calculation based on local utility rates

Target Market: Environmentally conscious homeowners, tech enthusiasts, cost-conscious families

Price Point: $149 for the hub + $29 per sensor (typical home needs 5-8 sensors)"""

    # Initialize all agent LLMs
    marketing_llm = ModelFactory.create(model_name, **config.get_specialist_kwargs())
    content_llm = ModelFactory.create(model_name, **config.get_specialist_kwargs())
    analyst_llm = ModelFactory.create(model_name, **config.get_specialist_kwargs())
    coordinator_llm = ModelFactory.create(model_name, **config.get_synthesizer_kwargs())

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"EXAMPLE 2: Product Launch Campaign (Parallel + Synthesis)")
        print(f"Model: {model_name}")
        print(f"{'=' * 80}\n")

    # --- Define Parallel Agent Nodes ---

    def marketing_strategist_node(state: ProductLaunchState) -> ProductLaunchState:
        """Marketing strategist develops overall strategy."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("AGENT 1: MARKETING STRATEGIST (Running in parallel)")
            print(f"{'─' * 80}")

        messages = [
            SystemMessage(content=config.marketing_strategist_prompt),
            HumanMessage(content=f"Develop a marketing strategy for:\n\n{state['product_description']}")
        ]

        response = marketing_llm.invoke(messages)
        strategy = response.content

        if verbose:
            print(f"\n📈 Marketing Strategy:")
            print(strategy)

        return {
            "marketing_strategy": strategy,
            "messages": messages + [response]
        }

    def content_writer_node(state: ProductLaunchState) -> ProductLaunchState:
        """Content writer creates promotional content ideas."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("AGENT 2: CONTENT WRITER (Running in parallel)")
            print(f"{'─' * 80}")

        messages = [
            SystemMessage(content=config.content_writer_prompt),
            HumanMessage(content=f"Create compelling content ideas for:\n\n{state['product_description']}")
        ]

        response = content_llm.invoke(messages)
        content = response.content

        if verbose:
            print(f"\n✍️ Content Ideas:")
            print(content)

        return {
            "content_ideas": content,
            "messages": messages + [response]
        }

    def data_analyst_node(state: ProductLaunchState) -> ProductLaunchState:
        """Data analyst provides market analysis."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("AGENT 3: DATA ANALYST (Running in parallel)")
            print(f"{'─' * 80}")

        messages = [
            SystemMessage(content=config.data_analyst_prompt),
            HumanMessage(content=f"Analyze the market for:\n\n{state['product_description']}")
        ]

        response = analyst_llm.invoke(messages)
        analysis = response.content

        if verbose:
            print(f"\n📊 Market Analysis:")
            print(analysis)

        return {
            "market_analysis": analysis,
            "messages": messages + [response]
        }

    def coordinator_node(state: ProductLaunchState) -> ProductLaunchState:
        """Coordinator synthesizes all inputs into unified launch plan."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("AGENT 4: COORDINATOR (Synthesizing all inputs)")
            print(f"{'─' * 80}")

        messages = [
            SystemMessage(content=config.coordinator_prompt),
            HumanMessage(content=f"""Create a comprehensive product launch plan by integrating:

Marketing Strategy:
{state['marketing_strategy']}

Content Ideas:
{state['content_ideas']}

Market Analysis:
{state['market_analysis']}

Product:
{state['product_description']}

Synthesize these into a cohesive, actionable launch plan.""")
        ]

        response = coordinator_llm.invoke(messages)
        plan = response.content

        if verbose:
            print(f"\n🚀 Final Launch Plan:")
            print(plan)

        return {
            **state,
            "launch_plan": plan,
            "messages": state["messages"] + messages + [response]
        }

    # --- Build Parallel + Synthesis Workflow ---
    workflow = StateGraph(ProductLaunchState)

    # Add all nodes
    workflow.add_node("marketing", marketing_strategist_node)
    workflow.add_node("content", content_writer_node)
    workflow.add_node("analyst", data_analyst_node)
    workflow.add_node("coordinator", coordinator_node)

    # Parallel execution: all three specialists run simultaneously
    workflow.set_entry_point("marketing")
    workflow.set_entry_point("content")
    workflow.set_entry_point("analyst")

    # All parallel agents feed into coordinator
    workflow.add_edge("marketing", "coordinator")
    workflow.add_edge("content", "coordinator")
    workflow.add_edge("analyst", "coordinator")
    workflow.add_edge("coordinator", END)

    # Compile and run
    app = workflow.compile()

    initial_state = {
        "product_description": product_description,
        "marketing_strategy": "",
        "content_ideas": "",
        "market_analysis": "",
        "launch_plan": "",
        "messages": []
    }

    final_state = app.invoke(initial_state)

    result = {
        "product_description": product_description[:200] + "...",
        "marketing_strategy": final_state["marketing_strategy"],
        "content_ideas": final_state["content_ideas"],
        "market_analysis": final_state["market_analysis"],
        "launch_plan": final_state["launch_plan"],
        "workflow_type": "parallel_synthesis",
        "num_agents": 4
    }

    if verbose:
        print(f"\n{'═' * 80}")
        print("PARALLEL + SYNTHESIS WORKFLOW COMPLETE")
        print(f"{'═' * 80}\n")

    # Write results
    writer = create_writer("07_multi_agent_collab")
    log_path = writer.write_result(
        result=result,
        model_name=model_name,
        input_data={"product_description": product_description[:200] + "..."},
        metadata={
            "example": "product_launch",
            "workflow_type": "parallel_synthesis",
            "num_agents": 4
        }
    )

    if verbose:
        print(f"Results written to: {log_path}\n")

    return result


# ============================================================================
# EXAMPLE 3: CODE REVIEW SYSTEM (MULTI-PERSPECTIVE DEBATE)
# ============================================================================

class CodeReviewState(TypedDict):
    """State for multi-perspective code review workflow with iterative refinement."""
    code_snippet: str  # Input code to review

    # Round 1: Initial independent reviews
    security_review: str  # Initial security perspective
    performance_review: str  # Initial performance perspective
    quality_review: str  # Initial code quality perspective

    # Round 2: Cross-review commentary and refined reviews
    security_cross_review: str  # Security's comments on other reviews
    performance_cross_review: str  # Performance's comments on other reviews
    quality_cross_review: str  # Quality's comments on other reviews

    security_review_refined: str  # Refined security review after cross-review
    performance_review_refined: str  # Refined performance review after cross-review
    quality_review_refined: str  # Refined quality review after cross-review

    # Round 3: Final synthesis
    synthesized_review: str  # Final integrated review
    messages: Annotated[list, operator.add]


def run_code_review(
    model_name: str = "gpt-4o",
    code_snippet: str = None,
    config: MultiAgentCollabConfig = None,
    verbose: bool = True
):
    """
    Run the multi-perspective code review workflow with iterative refinement.

    This demonstrates true multi-agent collaboration with inter-agent communication:

    Round 1 - Independent Reviews (Parallel):
    1. Security Reviewer: Identifies security issues
    2. Performance Reviewer: Identifies performance issues
    3. Quality Reviewer: Identifies code quality issues

    Round 2 - Cross-Review & Refinement (Collaborative):
    Each reviewer:
    - Sees other reviewers' findings
    - Identifies conflicts, synergies, or trade-offs
    - Updates their own review based on peer insights

    Round 3 - Final Synthesis:
    - Synthesizer creates prioritized recommendations from refined reviews

    Args:
        model_name: Model to use for all agents
        code_snippet: Code to review
        config: Pattern configuration
        verbose: Whether to print intermediate results

    Returns:
        Dictionary containing all reviews and final synthesis
    """
    config = config or MultiAgentCollabConfig()

    if code_snippet is None:
        code_snippet = """
def process_user_data(user_id, data):
    # Connect to database
    conn = mysql.connector.connect(
        host="localhost",
        user="admin",
        password="admin123",
        database="users"
    )
    cursor = conn.cursor()

    # Get user info
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    user = cursor.fetchone()

    # Process all data items
    results = []
    for item in data:
        # Heavy processing for each item
        processed = json.loads(json.dumps(item))
        result = {
            'id': item['id'],
            'value': item['value'] * 2,
            'timestamp': datetime.now()
        }
        results.append(result)

    # Update database
    for result in results:
        update_query = f"UPDATE users SET last_value = {result['value']} WHERE id = {user_id}"
        cursor.execute(update_query)

    conn.commit()
    cursor.close()
    conn.close()

    return results
"""

    # Initialize all reviewer LLMs
    security_llm = ModelFactory.create(model_name, **config.get_specialist_kwargs())
    performance_llm = ModelFactory.create(model_name, **config.get_specialist_kwargs())
    quality_llm = ModelFactory.create(model_name, **config.get_specialist_kwargs())
    synthesizer_llm = ModelFactory.create(model_name, **config.get_synthesizer_kwargs())

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"EXAMPLE 3: Code Review System (Multi-Agent Collaboration)")
        print(f"Model: {model_name}")
        print(f"3-Round Workflow: Independent Reviews → Cross-Review & Refinement → Synthesis")
        print(f"{'=' * 80}\n")

    # --- Define Reviewer Nodes ---

    def security_reviewer_node(state: CodeReviewState) -> CodeReviewState:
        """Security expert reviews code for vulnerabilities."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("REVIEWER 1: SECURITY EXPERT")
            print(f"{'─' * 80}")

        messages = [
            SystemMessage(content=config.security_reviewer_prompt),
            HumanMessage(content=f"Review this code for security issues:\n\n```python{state['code_snippet']}\n```")
        ]

        response = security_llm.invoke(messages)
        review = response.content

        if verbose:
            print(f"\n🔒 Security Review:")
            print(review)

        return {
            "security_review": review,
            "messages": messages + [response]
        }

    def performance_reviewer_node(state: CodeReviewState) -> CodeReviewState:
        """Performance expert reviews code for optimization opportunities."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("REVIEWER 2: PERFORMANCE EXPERT")
            print(f"{'─' * 80}")

        messages = [
            SystemMessage(content=config.performance_reviewer_prompt),
            HumanMessage(content=f"Review this code for performance issues:\n\n```python{state['code_snippet']}\n```")
        ]

        response = performance_llm.invoke(messages)
        review = response.content

        if verbose:
            print(f"\n⚡ Performance Review:")
            print(review)

        return {
            "performance_review": review,
            "messages": messages + [response]
        }

    def quality_reviewer_node(state: CodeReviewState) -> CodeReviewState:
        """Quality expert reviews code for best practices."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("REVIEWER 3: CODE QUALITY EXPERT")
            print(f"{'─' * 80}")

        messages = [
            SystemMessage(content=config.quality_reviewer_prompt),
            HumanMessage(content=f"Review this code for quality and best practices:\n\n```python{state['code_snippet']}\n```")
        ]

        response = quality_llm.invoke(messages)
        review = response.content

        if verbose:
            print(f"\n✨ Quality Review:")
            print(review)

        return {
            "quality_review": review,
            "messages": messages + [response]
        }

    # --- Round 2: Cross-Review & Refinement Nodes ---

    def security_cross_review_node(state: CodeReviewState) -> CodeReviewState:
        """Security reviewer sees other reviews and refines their own."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("ROUND 2: SECURITY EXPERT (Cross-Review & Refinement)")
            print(f"{'─' * 80}")

        messages = [
            SystemMessage(content=config.security_reviewer_prompt + """

You are now in the cross-review phase. Review your colleagues' findings and:
1. Identify any conflicts or synergies with your security recommendations
2. Note if performance or quality suggestions introduce security concerns
3. Update your security review to address these insights
4. Provide brief commentary on the collaboration"""),
            HumanMessage(content=f"""Original Code:
```python{state['code_snippet']}
```

YOUR Initial Security Review:
{state['security_review']}

PERFORMANCE REVIEWER's Findings:
{state['performance_review']}

QUALITY REVIEWER's Findings:
{state['quality_review']}

Based on your colleagues' reviews:
1. Provide commentary on synergies, conflicts, or trade-offs
2. Update your security review to create an integrated perspective""")
        ]

        response = security_llm.invoke(messages)
        full_response = response.content

        # Parse response to extract cross-review commentary and refined review
        if "CROSS-REVIEW COMMENTARY:" in full_response and "REFINED SECURITY REVIEW:" in full_response:
            parts = full_response.split("REFINED SECURITY REVIEW:")
            cross_review = parts[0].replace("CROSS-REVIEW COMMENTARY:", "").strip()
            refined_review = parts[1].strip()
        else:
            # Fallback: treat entire response as refined review
            cross_review = "Reviewed peer findings and integrated insights."
            refined_review = full_response

        if verbose:
            print(f"\n💬 Cross-Review Commentary:")
            print(cross_review[:300] + "..." if len(cross_review) > 300 else cross_review)
            print(f"\n🔒 Refined Security Review:")
            print(refined_review[:300] + "..." if len(refined_review) > 300 else refined_review)

        return {
            "security_cross_review": cross_review,
            "security_review_refined": refined_review,
            "messages": messages + [response]
        }

    def performance_cross_review_node(state: CodeReviewState) -> CodeReviewState:
        """Performance reviewer sees other reviews and refines their own."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("ROUND 2: PERFORMANCE EXPERT (Cross-Review & Refinement)")
            print(f"{'─' * 80}")

        messages = [
            SystemMessage(content=config.performance_reviewer_prompt + """

You are now in the cross-review phase. Review your colleagues' findings and:
1. Identify any conflicts or synergies with your performance recommendations
2. Note if security or quality suggestions impact performance
3. Update your performance review to address these insights
4. Provide brief commentary on the collaboration"""),
            HumanMessage(content=f"""Original Code:
```python{state['code_snippet']}
```

YOUR Initial Performance Review:
{state['performance_review']}

SECURITY REVIEWER's Findings:
{state['security_review']}

QUALITY REVIEWER's Findings:
{state['quality_review']}

Based on your colleagues' reviews:
1. Provide commentary on synergies, conflicts, or trade-offs
2. Update your performance review to create an integrated perspective""")
        ]

        response = performance_llm.invoke(messages)
        full_response = response.content

        # Parse response
        if "CROSS-REVIEW COMMENTARY:" in full_response and "REFINED PERFORMANCE REVIEW:" in full_response:
            parts = full_response.split("REFINED PERFORMANCE REVIEW:")
            cross_review = parts[0].replace("CROSS-REVIEW COMMENTARY:", "").strip()
            refined_review = parts[1].strip()
        else:
            cross_review = "Reviewed peer findings and integrated insights."
            refined_review = full_response

        if verbose:
            print(f"\n💬 Cross-Review Commentary:")
            print(cross_review[:300] + "..." if len(cross_review) > 300 else cross_review)
            print(f"\n⚡ Refined Performance Review:")
            print(refined_review[:300] + "..." if len(refined_review) > 300 else refined_review)

        return {
            "performance_cross_review": cross_review,
            "performance_review_refined": refined_review,
            "messages": messages + [response]
        }

    def quality_cross_review_node(state: CodeReviewState) -> CodeReviewState:
        """Quality reviewer sees other reviews and refines their own."""
        if verbose:
            print(f"\n{'─' * 80}")
            print("ROUND 2: CODE QUALITY EXPERT (Cross-Review & Refinement)")
            print(f"{'─' * 80}")

        messages = [
            SystemMessage(content=config.quality_reviewer_prompt + """

You are now in the cross-review phase. Review your colleagues' findings and:
1. Identify any conflicts or synergies with your quality recommendations
2. Note how security and performance concerns affect code quality
3. Update your quality review to address these insights
4. Provide brief commentary on the collaboration"""),
            HumanMessage(content=f"""Original Code:
```python{state['code_snippet']}
```

YOUR Initial Quality Review:
{state['quality_review']}

SECURITY REVIEWER's Findings:
{state['security_review']}

PERFORMANCE REVIEWER's Findings:
{state['performance_review']}

Based on your colleagues' reviews:
1. Provide commentary on synergies, conflicts, or trade-offs
2. Update your quality review to create an integrated perspective""")
        ]

        response = quality_llm.invoke(messages)
        full_response = response.content

        # Parse response
        if "CROSS-REVIEW COMMENTARY:" in full_response and "REFINED QUALITY REVIEW:" in full_response:
            parts = full_response.split("REFINED QUALITY REVIEW:")
            cross_review = parts[0].replace("CROSS-REVIEW COMMENTARY:", "").strip()
            refined_review = parts[1].strip()
        else:
            cross_review = "Reviewed peer findings and integrated insights."
            refined_review = full_response

        if verbose:
            print(f"\n💬 Cross-Review Commentary:")
            print(cross_review[:300] + "..." if len(cross_review) > 300 else cross_review)
            print(f"\n✨ Refined Quality Review:")
            print(refined_review[:300] + "..." if len(refined_review) > 300 else refined_review)

        return {
            "quality_cross_review": cross_review,
            "quality_review_refined": refined_review,
            "messages": messages + [response]
        }

    def synthesizer_node(state: CodeReviewState) -> CodeReviewState:
        """Synthesizer integrates refined collaborative reviews into final recommendations."""
        if verbose:
            print(f"\n{'═' * 80}")
            print("ROUND 3: FINAL SYNTHESIS (Integrating Refined Reviews)")
            print(f"{'═' * 80}")

        messages = [
            SystemMessage(content=config.review_synthesizer_prompt),
            HumanMessage(content=f"""Synthesize the REFINED code reviews (after cross-review collaboration) into a comprehensive, prioritized review.

Note: These are refined reviews where each expert has already considered their colleagues' perspectives.

REFINED Security Review (after cross-review):
{state['security_review_refined']}

REFINED Performance Review (after cross-review):
{state['performance_review_refined']}

REFINED Quality Review (after cross-review):
{state['quality_review_refined']}

Original Code:
```python{state['code_snippet']}
```

Create a unified review with prioritized action items that reflects the collaborative refinement process.""")
        ]

        response = synthesizer_llm.invoke(messages)
        synthesis = response.content

        if verbose:
            print(f"\n📋 Final Synthesized Review:")
            print(synthesis)

        return {
            "synthesized_review": synthesis,
            "messages": messages + [response]
        }

    # --- Build 3-Round Collaborative Review Workflow ---
    workflow = StateGraph(CodeReviewState)

    # Round 1: Independent parallel reviews
    workflow.add_node("security", security_reviewer_node)
    workflow.add_node("performance", performance_reviewer_node)
    workflow.add_node("quality", quality_reviewer_node)

    # Round 2: Cross-review and refinement (parallel, after Round 1 completes)
    workflow.add_node("security_refine", security_cross_review_node)
    workflow.add_node("performance_refine", performance_cross_review_node)
    workflow.add_node("quality_refine", quality_cross_review_node)

    # Round 3: Final synthesis
    workflow.add_node("synthesizer", synthesizer_node)

    # Define workflow edges
    # Round 1: All initial reviewers run in parallel
    workflow.set_entry_point("security")
    workflow.set_entry_point("performance")
    workflow.set_entry_point("quality")

    # Round 1 → Round 2: Each reviewer feeds into their own refinement node
    workflow.add_edge("security", "security_refine")
    workflow.add_edge("performance", "performance_refine")
    workflow.add_edge("quality", "quality_refine")

    # Round 2 → Round 3: All refined reviews feed into synthesizer
    workflow.add_edge("security_refine", "synthesizer")
    workflow.add_edge("performance_refine", "synthesizer")
    workflow.add_edge("quality_refine", "synthesizer")

    workflow.add_edge("synthesizer", END)

    # Compile and run
    app = workflow.compile()

    initial_state = {
        "code_snippet": code_snippet,
        # Round 1 fields
        "security_review": "",
        "performance_review": "",
        "quality_review": "",
        # Round 2 fields
        "security_cross_review": "",
        "performance_cross_review": "",
        "quality_cross_review": "",
        "security_review_refined": "",
        "performance_review_refined": "",
        "quality_review_refined": "",
        # Round 3 field
        "synthesized_review": "",
        "messages": []
    }

    final_state = app.invoke(initial_state)

    result = {
        "code_snippet": code_snippet[:200] + "...",
        # Round 1: Initial reviews
        "security_review": final_state["security_review"],
        "performance_review": final_state["performance_review"],
        "quality_review": final_state["quality_review"],
        # Round 2: Refined reviews after collaboration
        "security_review_refined": final_state["security_review_refined"],
        "performance_review_refined": final_state["performance_review_refined"],
        "quality_review_refined": final_state["quality_review_refined"],
        # Round 3: Final synthesis
        "synthesized_review": final_state["synthesized_review"],
        "workflow_type": "collaborative_iterative_refinement",
        "num_reviewers": 3,
        "num_rounds": 3
    }

    if verbose:
        print(f"\n{'═' * 80}")
        print("3-ROUND COLLABORATIVE REVIEW COMPLETE")
        print("Round 1: Independent reviews → Round 2: Cross-review & refinement → Round 3: Synthesis")
        print(f"{'═' * 80}\n")

    # Write results
    writer = create_writer("07_multi_agent_collab")
    log_path = writer.write_result(
        result=result,
        model_name=model_name,
        input_data={"code_snippet": code_snippet[:200] + "..."},
        metadata={
            "example": "code_review_collaborative",
            "workflow_type": "collaborative_iterative_refinement",
            "num_reviewers": 3,
            "num_rounds": 3,
            "has_cross_review": True
        }
    )

    if verbose:
        print(f"Results written to: {log_path}\n")

    return result


# ============================================================================
# COMPARISON AND MAIN FUNCTIONS
# ============================================================================

def compare_models(
    model_names: list[str] = None,
    example: str = "research_analysis",
    config: MultiAgentCollabConfig = None
):
    """
    Compare multi-agent collaboration across multiple models.

    Args:
        model_names: List of model names to compare
        example: Which example to run ('research_analysis', 'product_launch', or 'code_review')
        config: Pattern configuration

    Returns:
        Dictionary mapping model names to their results
    """
    if model_names is None:
        model_names = [
            "gpt-4o",
            "claude-sonnet-4-5-20250929",
            "gpt-4o-mini",
        ]

    config = config or MultiAgentCollabConfig()

    example_functions = {
        "research_analysis": run_research_analysis,
        "product_launch": run_product_launch,
        "code_review": run_code_review
    }

    if example not in example_functions:
        raise ValueError(f"Unknown example: {example}. Choose from {list(example_functions.keys())}")

    run_func = example_functions[example]

    print(f"\n{'=' * 80}")
    print(f"Comparing Multi-Agent Collaboration ({example}) Across Models")
    print(f"{'=' * 80}\n")

    results = {}
    for model in model_names:
        print(f"\nTesting with {model}...")
        try:
            result = run_func(
                model_name=model,
                config=config,
                verbose=False
            )
            results[model] = result
            print(f"✓ Completed successfully")
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[model] = f"Error: {e}"

    # Write comparison results
    writer = create_writer("07_multi_agent_collab")
    log_path = writer.write_comparison(
        results=results,
        input_data={"example": example}
    )

    print(f"\n{'=' * 80}")
    print(f"Comparison results written to: {log_path}")
    print(f"{'=' * 80}\n")

    return results


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o"
    example_num = sys.argv[2] if len(sys.argv) > 2 else "1"

    print(f"\n{'=' * 80}")
    print(f"Multi-Agent Collaboration Pattern - Examples")
    print(f"Model: {model}")
    print(f"{'=' * 80}\n")

    if example_num == "1":
        print("\n" + "=" * 80)
        print("EXAMPLE 1: Research Paper Analysis (Sequential Pipeline)")
        print("=" * 80)
        run_research_analysis(model_name=model)

    elif example_num == "2":
        print("\n" + "=" * 80)
        print("EXAMPLE 2: Product Launch Campaign (Parallel + Synthesis)")
        print("=" * 80)
        run_product_launch(model_name=model)

    elif example_num == "3":
        print("\n" + "=" * 80)
        print("EXAMPLE 3: Code Review System (Multi-Perspective)")
        print("=" * 80)
        run_code_review(model_name=model)

    elif example_num == "all":
        # Run all three examples
        print("\n" + "=" * 80)
        print("RUNNING ALL EXAMPLES")
        print("=" * 80)

        print("\n" + "=" * 80)
        print("EXAMPLE 1: Research Paper Analysis")
        print("=" * 80)
        run_research_analysis(model_name=model)

        print("\n" + "=" * 80)
        print("EXAMPLE 2: Product Launch Campaign")
        print("=" * 80)
        run_product_launch(model_name=model)

        print("\n" + "=" * 80)
        print("EXAMPLE 3: Code Review System")
        print("=" * 80)
        run_code_review(model_name=model)

    else:
        print(f"Unknown example number: {example_num}")
        print("Usage: python run.py [model_name] [example_num]")
        print("  example_num: 1, 2, 3, or 'all'")
        print("  1 = Research Paper Analysis (Sequential)")
        print("  2 = Product Launch Campaign (Parallel)")
        print("  3 = Code Review System (Multi-Perspective)")
        print("  all = Run all examples")
