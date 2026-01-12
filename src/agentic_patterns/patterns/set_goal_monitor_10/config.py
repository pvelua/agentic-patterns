"""Configuration for the Goal Setting and Monitoring pattern."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GoalMonitorConfig:
    """Configuration for two-agent goal monitoring pattern."""

    # Developer agent (code generator) configuration
    default_developer_model: str = "gpt-4o"
    developer_temperature: float = 0.2  # More deterministic for code generation
    developer_max_tokens: int = 4000

    # Manager agent (code reviewer) configuration
    default_manager_model: str = "claude-sonnet-4-5-20250929"
    manager_temperature: float = 0.4  # More analytical for review
    manager_max_tokens: int = 3000

    # Iteration settings
    max_iterations: int = 4
    passing_grade: float = 85.0  # Stop if we reach this grade (0-100 scale)

    # Grading weights (must sum to 100)
    weight_requirements: int = 40  # Requirements coverage
    weight_quality: int = 30       # Code quality and structure
    weight_error_handling: int = 15  # Robustness and edge cases
    weight_documentation: int = 15   # Docstrings and clarity

    # Sample User Goals with Requirements
    sample_goals: Dict[str, Dict] = field(default_factory=lambda: {
        'api_client': {
            'title': 'REST API Client Builder',
            'description': 'Build a reusable REST API client class for making HTTP requests',
            'requirements': [
                'Support GET, POST, PUT, DELETE HTTP methods',
                'Include automatic retry logic with exponential backoff (max 3 retries)',
                'Implement rate limiting (configurable requests per second)',
                'Handle common HTTP errors (4xx, 5xx) with appropriate exceptions',
                'Support request/response logging for debugging',
                'Include timeout configuration (default 30 seconds)',
                'Parse JSON responses automatically',
                'Add custom headers support (authentication, user-agent, etc.)'
            ],
            'example_usage': '''
client = APIClient(base_url="https://api.example.com", rate_limit=10)
response = client.get("/users/123", headers={"Authorization": "Bearer token"})
data = client.post("/users", json={"name": "John", "email": "john@example.com"})
'''
        },
        'data_validator': {
            'title': 'Data Validation Library',
            'description': 'Create a flexible data validation library with custom rules',
            'requirements': [
                'Support validation rules: required, type checking, min/max length',
                'Support custom validation functions (callbacks)',
                'Return detailed error messages with field names and reasons',
                'Support nested object validation (dictionaries)',
                'Support list/array validation (validate each item)',
                'Provide a fluent API for chaining validators',
                'Include common validators: email, URL, phone number, date format',
                'Support conditional validation (validate field X only if Y is present)'
            ],
            'example_usage': '''
validator = Validator()
validator.field("email").required().email()
validator.field("age").required().integer().min(18).max(120)
validator.field("tags").list().min_length(1)

errors = validator.validate(data)
if errors:
    print(f"Validation failed: {errors}")
'''
        },
        'task_scheduler': {
            'title': 'Task Scheduling System',
            'description': 'Implement a task scheduler with priority queues and scheduling',
            'requirements': [
                'Support priority-based task execution (high, medium, low)',
                'Implement cron-like scheduling (run at specific times/intervals)',
                'Support one-time and recurring tasks',
                'Include task cancellation and rescheduling',
                'Handle task failures with retry logic',
                'Provide task status tracking (pending, running, completed, failed)',
                'Support task dependencies (task B runs after task A completes)',
                'Implement concurrent execution with configurable worker threads'
            ],
            'example_usage': '''
scheduler = TaskScheduler(workers=4)
scheduler.schedule_task(send_email, priority="high", run_at="09:00")
scheduler.schedule_recurring(cleanup_temp, interval="1h")
scheduler.schedule_with_dependencies(task_b, depends_on=[task_a])
scheduler.start()
'''
        }
    })

    # Developer Agent System Prompt
    developer_system_prompt: str = """You are an expert Python developer specializing in clean, production-ready code.

Your role:
1. Analyze user requirements carefully
2. Create a clear implementation plan
3. Write high-quality Python code that meets ALL requirements
4. Include proper error handling and edge cases
5. Add comprehensive docstrings and comments
6. Follow Python best practices (PEP 8, type hints, etc.)

When receiving feedback from the project manager:
- Read the review carefully and understand all issues
- Address EVERY point raised in the feedback
- Fix bugs and missing requirements
- Improve code quality and documentation
- Explain what changes you made

Code quality standards:
- Use type hints for function signatures
- Include docstrings for classes and public methods
- Handle errors gracefully with try-except blocks
- Validate inputs and handle edge cases
- Write modular, reusable code
- Keep functions focused (single responsibility)
- Use meaningful variable and function names"""

    # Developer Planning Prompt
    developer_planning_template: str = """You are tasked with implementing the following:

GOAL: {goal_title}
DESCRIPTION: {goal_description}

REQUIREMENTS:
{requirements_list}

EXAMPLE USAGE:
{example_usage}

Please create a detailed implementation plan including:
1. Main classes/functions needed
2. Key design decisions
3. How each requirement will be addressed
4. Potential challenges and solutions

Provide your plan in a clear, structured format."""

    # Developer Implementation Prompt
    developer_implementation_template: str = """Based on your plan, implement the complete solution in Python.

GOAL: {goal_title}
REQUIREMENTS:
{requirements_list}

YOUR IMPLEMENTATION PLAN:
{implementation_plan}

{feedback_section}

Provide ONLY the complete Python code. Include:
- All necessary imports
- Complete class/function implementations
- Comprehensive docstrings
- Error handling
- Example usage in comments or docstring

Do NOT include markdown code blocks, just raw Python code."""

    # Developer Improvement Prompt (when receiving feedback)
    developer_improvement_section: str = """
PROJECT MANAGER REVIEW (Iteration {iteration}):
Grade: {grade}/100

ASSESSMENT:
{assessment}

FEEDBACK:
{feedback}

Based on this review, improve your code to address ALL issues and increase the grade.
Focus especially on the low-scoring areas."""

    # Manager Agent System Prompt
    manager_system_prompt: str = """You are an expert project manager and code reviewer specializing in quality assurance.

Your role:
1. Monitor developer progress against user requirements
2. Review code quality, structure, and best practices
3. Assess error handling and edge cases
4. Evaluate documentation quality
5. Provide detailed, actionable feedback
6. Grade the work objectively

Grading criteria:
- Requirements Coverage (40%): How many requirements are fully met
- Code Quality (30%): Structure, readability, best practices, PEP 8
- Error Handling (15%): Robustness, edge cases, exception handling
- Documentation (15%): Docstrings, comments, clarity

Be constructive but honest. Identify both strengths and areas for improvement.
Provide specific examples and suggestions."""

    # Manager Review Prompt
    manager_review_template: str = """Review the following code implementation:

GOAL: {goal_title}
REQUIREMENTS:
{requirements_list}

ITERATION: {iteration}/{max_iterations}

DEVELOPER'S CODE:
```python
{code}
```

Evaluate this implementation based on:

1. REQUIREMENTS COVERAGE (40 points):
   - Check each requirement individually
   - Identify which requirements are met, partially met, or missing
   - Score: X/40

2. CODE QUALITY (30 points):
   - Structure and organization
   - Readability and maintainability
   - Python best practices (PEP 8, type hints, naming conventions)
   - Modularity and reusability
   - Score: X/30

3. ERROR HANDLING (15 points):
   - Exception handling
   - Edge case coverage
   - Input validation
   - Robustness
   - Score: X/15

4. DOCUMENTATION (15 points):
   - Docstrings for classes and methods
   - Comments where needed
   - Code clarity
   - Example usage
   - Score: X/15

Provide your review in this EXACT format:

GRADE: <total score 0-100>

REQUIREMENTS COVERAGE (<score>/40):
- Requirement 1: <Met/Partially Met/Not Met> - <brief explanation>
- Requirement 2: <Met/Partially Met/Not Met> - <brief explanation>
...

CODE QUALITY (<score>/30):
<2-3 sentences on code quality>

ERROR HANDLING (<score>/15):
<2-3 sentences on error handling>

DOCUMENTATION (<score>/15):
<2-3 sentences on documentation>

OVERALL ASSESSMENT:
<1-2 paragraphs summarizing strengths and weaknesses>

FEEDBACK FOR IMPROVEMENT:
1. <Specific actionable feedback>
2. <Specific actionable feedback>
3. <Specific actionable feedback>
..."""

    def get_developer_kwargs(self) -> dict:
        """Return parameters for developer agent."""
        return {
            'temperature': self.developer_temperature,
            'max_tokens': self.developer_max_tokens
        }

    def get_manager_kwargs(self) -> dict:
        """Return parameters for manager agent."""
        return {
            'temperature': self.manager_temperature,
            'max_tokens': self.manager_max_tokens
        }

    def parse_grade(self, review_text: str) -> float:
        """Extract numeric grade from manager's review.

        Args:
            review_text: Full review text from manager

        Returns:
            Grade as float (0-100), or 0 if not found
        """
        import re

        # Look for "GRADE: XX" pattern
        match = re.search(r'GRADE:\s*(\d+(?:\.\d+)?)', review_text, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # Fallback: look for "XX/100" pattern
        match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*100', review_text)
        if match:
            return float(match.group(1))

        return 0.0
