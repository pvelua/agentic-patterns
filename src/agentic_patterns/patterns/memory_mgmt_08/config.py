"""Configuration for the Memory Management pattern."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MemoryManagementConfig:
    """Configuration for memory management pattern with different memory types."""

    # LLM parameters for different use cases
    assistant_temperature: float = 0.3  # Balanced for assistant tasks
    support_temperature: float = 0.4    # Slightly higher for empathetic support
    tutor_temperature: float = 0.5      # Higher for creative teaching

    # Token limits
    assistant_max_tokens: int = 2000
    support_max_tokens: int = 2500
    tutor_max_tokens: int = 3000

    # Memory buffer sizes
    conversation_buffer_k: int = 10  # Keep last K messages in buffer
    max_semantic_facts: int = 50     # Maximum facts to maintain
    max_episodic_memories: int = 20  # Maximum episodes to remember
    max_procedural_rules: int = 30   # Maximum rules to maintain

    # Example 1: Personal AI Assistant (ConversationBufferMemory + Semantic Memory)
    # Demonstrates automated short-term memory + long-term fact storage

    assistant_system_prompt: str = """You are a helpful personal AI assistant with access to the user's preferences and context.

Your capabilities:
1. Remember conversation context from recent messages (short-term memory)
2. Recall user preferences, contacts, and personal facts (semantic memory)
3. Use this information to provide personalized, context-aware assistance

When helping the user:
- Reference relevant facts from their profile
- Maintain conversation continuity
- Provide personalized recommendations based on preferences
- Ask clarifying questions when needed

Be friendly, efficient, and proactive in using stored information."""

    # Default semantic memory for assistant (user facts/preferences)
    default_semantic_memory: Dict[str, str] = field(default_factory=lambda: {
        "preferred_meeting_time": "10 AM on weekdays",
        "contact_john_email": "john.smith@example.com",
        "contact_sarah_email": "sarah.jones@example.com",
        "favorite_restaurant": "The Green Olive (Italian)",
        "dietary_restrictions": "vegetarian",
        "home_address": "123 Main St, Springfield",
        "work_start_time": "9 AM",
        "lunch_break": "12:30 PM - 1:30 PM",
        "timezone": "EST",
        "commute_duration": "25 minutes"
    })

    # Example 2: Customer Support Agent (ChatMessageHistory + Episodic + Procedural)
    # Demonstrates manual memory management + experience and rule-based memory

    support_system_prompt: str = """You are an expert customer support agent for TechGadget Inc., a consumer electronics company.

Your knowledge includes:
1. Current conversation history (short-term memory)
2. Past support interactions and resolutions (episodic memory)
3. Company policies and troubleshooting procedures (procedural memory)

When assisting customers:
- Review past tickets to understand customer history
- Apply relevant company policies and procedures
- Follow established troubleshooting steps
- Escalate to supervisor when procedures dictate
- Learn from past successful resolutions

Be empathetic, professional, and solution-oriented."""

    # Default episodic memory for support agent (past tickets)
    default_episodic_memory: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "ticket_id": "T-2024-001",
            "date": "2024-12-15",
            "issue": "WiFi connectivity problems with SmartHome Hub",
            "resolution": "Reset device to factory settings and updated firmware to v2.1.5",
            "outcome": "Issue resolved, customer satisfied"
        },
        {
            "ticket_id": "T-2024-045",
            "date": "2024-12-20",
            "issue": "Battery draining quickly on fitness tracker",
            "resolution": "Disabled always-on display and background sync, recommended battery replacement",
            "outcome": "Partially resolved, sent replacement unit"
        },
        {
            "ticket_id": "T-2025-003",
            "date": "2025-01-02",
            "issue": "Unable to pair Bluetooth earbuds with phone",
            "resolution": "Cleared Bluetooth cache, re-paired devices, updated phone OS",
            "outcome": "Issue resolved"
        }
    ])

    # Default procedural memory for support agent (company policies/procedures)
    default_procedural_memory: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "rule_id": "POL-001",
            "category": "warranty",
            "rule": "Products under 1-year warranty: free replacement. Over 1 year: 50% discount on replacement."
        },
        {
            "rule_id": "POL-002",
            "category": "escalation",
            "rule": "Escalate to supervisor if: customer requests refund >$500, legal threat, or 3+ failed resolution attempts."
        },
        {
            "rule_id": "PROC-001",
            "category": "troubleshooting",
            "rule": "WiFi issues: 1) Check router, 2) Restart device, 3) Forget/re-add network, 4) Factory reset, 5) Firmware update."
        },
        {
            "rule_id": "PROC-002",
            "category": "troubleshooting",
            "rule": "Battery issues: 1) Check battery health, 2) Disable power-hungry features, 3) Software update, 4) Battery replacement."
        },
        {
            "rule_id": "POL-003",
            "category": "returns",
            "rule": "30-day return policy for unopened products. Opened products: 15-day return with 20% restocking fee."
        },
        {
            "rule_id": "PROC-003",
            "category": "troubleshooting",
            "rule": "Bluetooth pairing: 1) Enable pairing mode, 2) Clear paired devices, 3) Clear Bluetooth cache, 4) OS update, 5) Reset device."
        }
    ])

    # Example 3: Adaptive Learning Tutor (All Memory Types)
    # Demonstrates comprehensive memory management for personalized education

    tutor_system_prompt: str = """You are an adaptive learning tutor specializing in mathematics for middle school students.

Your memory systems include:
1. Current lesson conversation (short-term memory)
2. Mathematical concepts and knowledge base (semantic memory)
3. Student's learning history, mistakes, and progress (episodic memory)
4. Teaching strategies and pedagogical rules (procedural memory)

When tutoring:
- Adapt difficulty based on student's past performance
- Reference previous lessons and common mistakes
- Apply appropriate teaching strategies (hints vs. direct answers)
- Build on concepts the student has mastered
- Identify knowledge gaps from episodic history

Be encouraging, patient, and adjust your teaching style to the student's needs."""

    # Default semantic memory for tutor (mathematical knowledge)
    default_tutor_semantic_memory: Dict[str, str] = field(default_factory=lambda: {
        "algebra_basics": "Variables represent unknown values; equations show relationships between variables",
        "fractions": "Part of a whole; numerator/denominator; must find common denominator for addition/subtraction",
        "percentages": "Parts per hundred; multiply by decimal form (e.g., 25% = 0.25)",
        "geometry_area": "Rectangle: length × width; Triangle: ½ × base × height; Circle: π × radius²",
        "order_of_operations": "PEMDAS: Parentheses, Exponents, Multiplication/Division (left to right), Addition/Subtraction (left to right)",
        "pythagorean_theorem": "For right triangles: a² + b² = c², where c is hypotenuse",
        "linear_equations": "Form: y = mx + b, where m is slope and b is y-intercept",
        "probability": "P(event) = favorable outcomes / total possible outcomes"
    })

    # Default episodic memory for tutor (student's learning history)
    default_tutor_episodic_memory: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "session_id": "S-001",
            "date": "2025-01-05",
            "topic": "Fractions - Addition",
            "performance": "Struggled with finding common denominators, but improved with practice",
            "mistakes": "Forgot to convert to common denominator in 3/5 problems",
            "mastery_level": "60%"
        },
        {
            "session_id": "S-002",
            "date": "2025-01-06",
            "topic": "Fractions - Multiplication",
            "performance": "Excellent! Understood multiply-across concept quickly",
            "mistakes": "None significant",
            "mastery_level": "90%"
        },
        {
            "session_id": "S-003",
            "date": "2025-01-07",
            "topic": "Order of Operations",
            "performance": "Confused about parentheses priority, needs more practice",
            "mistakes": "Did multiplication before solving parentheses in 2/4 problems",
            "mastery_level": "50%"
        }
    ])

    # Default procedural memory for tutor (teaching strategies)
    default_tutor_procedural_memory: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "rule_id": "TEACH-001",
            "category": "scaffolding",
            "rule": "If student struggles (mastery < 60%), break problem into smaller steps with hints"
        },
        {
            "rule_id": "TEACH-002",
            "category": "scaffolding",
            "rule": "If student excels (mastery > 85%), introduce more challenging variations"
        },
        {
            "rule_id": "TEACH-003",
            "category": "feedback",
            "rule": "For repeated mistakes: identify misconception, explain why error occurred, provide similar practice problem"
        },
        {
            "rule_id": "TEACH-004",
            "category": "review",
            "rule": "Start each session reviewing previous topic's key concept before introducing new material"
        },
        {
            "rule_id": "TEACH-005",
            "category": "engagement",
            "rule": "Use real-world examples related to student's interests to illustrate abstract concepts"
        },
        {
            "rule_id": "TEACH-006",
            "category": "assessment",
            "rule": "After explaining concept, ask student to solve similar problem independently to verify understanding"
        }
    ])

    def get_assistant_kwargs(self) -> dict:
        """Return parameters for assistant agent."""
        return {
            'temperature': self.assistant_temperature,
            'max_tokens': self.assistant_max_tokens
        }

    def get_support_kwargs(self) -> dict:
        """Return parameters for support agent."""
        return {
            'temperature': self.support_temperature,
            'max_tokens': self.support_max_tokens
        }

    def get_tutor_kwargs(self) -> dict:
        """Return parameters for tutor agent."""
        return {
            'temperature': self.tutor_temperature,
            'max_tokens': self.tutor_max_tokens
        }

    # Example 4: Personal Finance Advisor (InMemoryStore - Production Pattern)
    # Demonstrates: ConversationBufferMemory + InMemoryStore for all long-term memory types

    advisor_temperature: float = 0.3  # Balanced for financial advice
    advisor_max_tokens: int = 2500

    advisor_system_prompt: str = """You are an expert personal finance advisor helping clients make informed financial decisions.

Your memory systems include:
1. Current conversation context (short-term memory)
2. Financial knowledge base - tax rules, investment principles, financial concepts (semantic memory via InMemoryStore)
3. Client's financial history - past decisions, transactions, advice given (episodic memory via InMemoryStore)
4. Financial strategies and advisory protocols (procedural memory via InMemoryStore)

When advising clients:
- Review their financial history to provide consistent, informed advice
- Apply relevant tax rules and financial principles
- Follow established investment strategies and risk assessment protocols
- Reference past decisions to ensure continuity
- Provide personalized recommendations based on their profile

Be professional, prudent, and always consider the client's long-term financial health."""

    # Default semantic memory for advisor (financial knowledge) - will be stored in InMemoryStore
    default_advisor_semantic_memory: Dict[str, str] = field(default_factory=lambda: {
        "tax_401k": "401(k) contributions are pre-tax, reducing current taxable income. Max contribution $23,000 (2024), $30,500 if age 50+",
        "tax_roth_ira": "Roth IRA contributions are post-tax. Earnings grow tax-free. Income limits apply. Max contribution $7,000 (2024), $8,000 if age 50+",
        "emergency_fund": "Recommended emergency fund: 3-6 months of expenses in liquid, accessible accounts",
        "asset_allocation": "Asset allocation should match risk tolerance and time horizon. Common rule: stocks% = 100 - age",
        "diversification": "Diversify across asset classes (stocks, bonds, real estate) and within each class to reduce risk",
        "compound_interest": "Power of compound interest: earlier investments have more time to grow exponentially",
        "inflation": "Inflation erodes purchasing power over time. Investments should aim to beat inflation rate (typically 2-3%)",
        "risk_return": "Higher potential returns generally come with higher risk. Balance based on goals and timeline"
    })

    # Default episodic memory for advisor (client's financial history) - will be stored in InMemoryStore
    default_advisor_episodic_memory: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "session_id": "FIN-001",
            "date": "2024-11-15",
            "topic": "Retirement planning discussion",
            "decision": "Started contributing 15% to 401(k), increased from 5%",
            "context": "Client age 35, goal to retire at 65, moderate risk tolerance",
            "outcome": "Set up automatic contribution increase"
        },
        {
            "session_id": "FIN-002",
            "date": "2024-12-01",
            "topic": "Emergency fund review",
            "decision": "Opened high-yield savings account with $10,000 initial deposit",
            "context": "Goal: build to $20,000 (6 months expenses)",
            "outcome": "On track, adding $500/month"
        },
        {
            "session_id": "FIN-003",
            "date": "2024-12-20",
            "topic": "Tax planning for year-end",
            "decision": "Maximized 401(k) contribution for tax year, considered Roth conversion",
            "context": "High income year, seeking tax optimization",
            "outcome": "Deferred Roth conversion to next year due to income spike"
        },
        {
            "session_id": "FIN-004",
            "date": "2025-01-05",
            "topic": "Investment portfolio rebalancing",
            "decision": "Rebalanced to 70% stocks, 30% bonds to match risk tolerance",
            "context": "Portfolio had drifted to 80% stocks due to market gains",
            "outcome": "Sold some stock funds, bought bond index fund"
        }
    ])

    # Default procedural memory for advisor (strategies and protocols) - will be stored in InMemoryStore
    default_advisor_procedural_memory: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "rule_id": "STRAT-001",
            "category": "investment_strategy",
            "rule": "For retirement accounts: Prioritize 401(k) up to employer match, then max Roth IRA, then max 401(k)"
        },
        {
            "rule_id": "STRAT-002",
            "category": "risk_assessment",
            "rule": "Risk tolerance assessment: Conservative if <5yr horizon, Moderate if 5-15yr, Aggressive if >15yr to goal"
        },
        {
            "rule_id": "STRAT-003",
            "category": "emergency_fund",
            "rule": "Emergency fund priority: 1) $1000 starter fund, 2) Pay off high-interest debt, 3) Build to 3-6 months expenses"
        },
        {
            "rule_id": "STRAT-004",
            "category": "rebalancing",
            "rule": "Rebalance portfolio when allocation drifts >5% from target, or at minimum annually"
        },
        {
            "rule_id": "STRAT-005",
            "category": "tax_optimization",
            "rule": "Tax-loss harvesting: Sell losing positions to offset capital gains, maintain similar exposure with different securities"
        },
        {
            "rule_id": "PROTO-001",
            "category": "advisory_protocol",
            "rule": "Always review client's complete financial history before major recommendations to ensure consistency"
        },
        {
            "rule_id": "PROTO-002",
            "category": "advisory_protocol",
            "rule": "For changes to asset allocation >10%, explain rationale and confirm client understanding before proceeding"
        }
    ])

    def get_advisor_kwargs(self) -> dict:
        """Return parameters for advisor agent."""
        return {
            'temperature': self.advisor_temperature,
            'max_tokens': self.advisor_max_tokens
        }

    def get_model_kwargs(self) -> dict:
        """Return default parameters for ModelFactory.create() (backward compatibility)"""
        return self.get_assistant_kwargs()
