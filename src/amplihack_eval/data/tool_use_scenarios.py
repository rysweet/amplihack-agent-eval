"""Tool use scenarios for L13 evaluation.

Defines 20+ scenarios where an agent must select the correct tool(s) from a set
of available tools. Scenarios span memory search, fact storage, knowledge
explanation, gap finding, and verification. Includes multi-step tool chains.

Philosophy: Data-driven scenario definition, separates tool use content from
scoring logic. Each scenario is self-contained and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolDefinition:
    """A tool available to the agent."""

    name: str
    description: str
    parameters: list[str] = field(default_factory=list)


@dataclass
class ToolUseScenario:
    """A scenario requiring the agent to select and chain the right tools."""

    scenario_id: str
    task_description: str
    available_tools: list[ToolDefinition]
    expected_tool_sequence: list[str]  # Ordered list of tool names
    incorrect_alternatives: list[list[str]]  # Wrong sequences that look plausible
    domain: str  # memory_search, fact_storage, explanation, gap_finding, verification
    difficulty: str  # single, chain, complex_chain
    rationale: str  # Why the expected sequence is correct


# ── Standard tool palette ──────────────────────────────────────────────

MEMORY_SEARCH = ToolDefinition(
    name="memory_search",
    description="Search stored memory for facts matching a query",
    parameters=["query", "max_results"],
)

FACT_STORE = ToolDefinition(
    name="fact_store",
    description="Store a new fact or update an existing one in memory",
    parameters=["fact_text", "source", "timestamp"],
)

KNOWLEDGE_EXPLAIN = ToolDefinition(
    name="knowledge_explain",
    description="Generate a natural language explanation of stored knowledge",
    parameters=["topic", "detail_level"],
)

GAP_FINDER = ToolDefinition(
    name="gap_finder",
    description="Identify gaps or missing information in current knowledge",
    parameters=["domain", "expected_coverage"],
)

FACT_VERIFY = ToolDefinition(
    name="fact_verify",
    description="Cross-reference a fact against multiple stored sources",
    parameters=["fact_text", "confidence_threshold"],
)

TEMPORAL_QUERY = ToolDefinition(
    name="temporal_query",
    description="Query facts with time-range filtering",
    parameters=["query", "start_date", "end_date"],
)

SUMMARIZE = ToolDefinition(
    name="summarize",
    description="Summarize a collection of facts into a concise overview",
    parameters=["topic", "max_length"],
)

COMPARE = ToolDefinition(
    name="compare",
    description="Compare two entities or time periods on specified dimensions",
    parameters=["entity_a", "entity_b", "dimensions"],
)

CLASSIFY = ToolDefinition(
    name="classify",
    description="Classify a piece of information by category, urgency, or domain",
    parameters=["text", "taxonomy"],
)

ALERT = ToolDefinition(
    name="alert",
    description="Send a notification about a critical finding",
    parameters=["message", "severity"],
)

ALL_TOOLS = [
    MEMORY_SEARCH,
    FACT_STORE,
    KNOWLEDGE_EXPLAIN,
    GAP_FINDER,
    FACT_VERIFY,
    TEMPORAL_QUERY,
    SUMMARIZE,
    COMPARE,
    CLASSIFY,
    ALERT,
]

# ── Scenarios ──────────────────────────────────────────────────────────

SCENARIO_01 = ToolUseScenario(
    scenario_id="TU01",
    task_description="Find all facts about Project Atlas deadlines.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["memory_search"],
    incorrect_alternatives=[
        ["knowledge_explain"],
        ["gap_finder"],
        ["temporal_query"],
    ],
    domain="memory_search",
    difficulty="single",
    rationale="Simple fact retrieval requires only memory_search with a targeted query.",
)

SCENARIO_02 = ToolUseScenario(
    scenario_id="TU02",
    task_description="Store the fact that Project Atlas deadline moved to September 1.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["fact_store"],
    incorrect_alternatives=[
        ["memory_search"],
        ["memory_search", "fact_store"],
    ],
    domain="fact_storage",
    difficulty="single",
    rationale="Direct storage of a new fact only requires fact_store.",
)

SCENARIO_03 = ToolUseScenario(
    scenario_id="TU03",
    task_description="Explain the current security incident response procedure to a new team member.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["memory_search", "knowledge_explain"],
    incorrect_alternatives=[
        ["knowledge_explain"],
        ["summarize"],
        ["memory_search", "summarize"],
    ],
    domain="explanation",
    difficulty="chain",
    rationale="Must first retrieve procedure facts, then generate an explanation.",
)

SCENARIO_04 = ToolUseScenario(
    scenario_id="TU04",
    task_description="Check if the reported server uptime of 99.99% is consistent with stored monitoring data.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["memory_search", "fact_verify"],
    incorrect_alternatives=[
        ["fact_verify"],
        ["memory_search"],
        ["memory_search", "compare"],
    ],
    domain="verification",
    difficulty="chain",
    rationale="Must retrieve monitoring data first, then verify the claimed uptime against it.",
)

SCENARIO_05 = ToolUseScenario(
    scenario_id="TU05",
    task_description="What information are we missing about the new hire's onboarding status?",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["memory_search", "gap_finder"],
    incorrect_alternatives=[
        ["gap_finder"],
        ["memory_search"],
        ["memory_search", "knowledge_explain"],
    ],
    domain="gap_finding",
    difficulty="chain",
    rationale="Must search existing facts to establish baseline, then identify gaps.",
)

SCENARIO_06 = ToolUseScenario(
    scenario_id="TU06",
    task_description="Compare Q1 and Q2 sales performance and explain the trend.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["temporal_query", "compare", "knowledge_explain"],
    incorrect_alternatives=[
        ["compare"],
        ["memory_search", "compare"],
        ["temporal_query", "summarize"],
    ],
    domain="explanation",
    difficulty="complex_chain",
    rationale="Need temporal data for both quarters, then comparison, then explanation of the trend.",
)

SCENARIO_07 = ToolUseScenario(
    scenario_id="TU07",
    task_description="A security alert reports a brute force attack from 192.168.1.45. Verify if this IP has been seen before and classify the threat.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["memory_search", "fact_verify", "classify"],
    incorrect_alternatives=[
        ["classify"],
        ["memory_search", "classify"],
        ["memory_search", "alert"],
    ],
    domain="verification",
    difficulty="complex_chain",
    rationale="Search for prior IP activity, verify against threat intel, then classify severity.",
)

SCENARIO_08 = ToolUseScenario(
    scenario_id="TU08",
    task_description="Summarize all infrastructure changes made in the last 30 days.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["temporal_query", "summarize"],
    incorrect_alternatives=[
        ["memory_search", "summarize"],
        ["summarize"],
        ["temporal_query", "knowledge_explain"],
    ],
    domain="memory_search",
    difficulty="chain",
    rationale="Time-bounded query requires temporal_query, then summarize for overview.",
)

SCENARIO_09 = ToolUseScenario(
    scenario_id="TU09",
    task_description="A new vulnerability CVE-2026-1234 is reported. Store it, check if any of our systems are affected, and send an alert if critical.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["fact_store", "memory_search", "classify", "alert"],
    incorrect_alternatives=[
        ["fact_store", "alert"],
        ["memory_search", "alert"],
        ["fact_store", "memory_search", "alert"],
    ],
    domain="verification",
    difficulty="complex_chain",
    rationale="Store the CVE, search for affected systems, classify severity, then alert if critical.",
)

SCENARIO_10 = ToolUseScenario(
    scenario_id="TU10",
    task_description="What is Sarah Chen's current role?",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["memory_search"],
    incorrect_alternatives=[
        ["temporal_query"],
        ["memory_search", "fact_verify"],
    ],
    domain="memory_search",
    difficulty="single",
    rationale="Simple lookup of a current fact requires only memory_search.",
)

SCENARIO_11 = ToolUseScenario(
    scenario_id="TU11",
    task_description="Update the record: Sarah Chen has been promoted from Senior Engineer to Engineering Director.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["memory_search", "fact_store"],
    incorrect_alternatives=[
        ["fact_store"],
        ["memory_search", "fact_verify", "fact_store"],
    ],
    domain="fact_storage",
    difficulty="chain",
    rationale="Must search to find existing record, then store the updated role.",
)

SCENARIO_12 = ToolUseScenario(
    scenario_id="TU12",
    task_description="Verify that the budget figures in the Q3 report match what we have stored.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["memory_search", "fact_verify"],
    incorrect_alternatives=[
        ["fact_verify"],
        ["memory_search", "compare"],
        ["temporal_query", "fact_verify"],
    ],
    domain="verification",
    difficulty="chain",
    rationale="Retrieve stored budget data, then verify against the report figures.",
)

SCENARIO_13 = ToolUseScenario(
    scenario_id="TU13",
    task_description="Identify what we don't know about the competitor's upcoming product launch.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["memory_search", "gap_finder"],
    incorrect_alternatives=[
        ["gap_finder"],
        ["memory_search", "knowledge_explain"],
        ["memory_search", "summarize"],
    ],
    domain="gap_finding",
    difficulty="chain",
    rationale="Must check what we already know, then identify specific gaps.",
)

SCENARIO_14 = ToolUseScenario(
    scenario_id="TU14",
    task_description="Explain how the database migration process works, including any recent changes.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["memory_search", "temporal_query", "knowledge_explain"],
    incorrect_alternatives=[
        ["knowledge_explain"],
        ["memory_search", "knowledge_explain"],
        ["temporal_query", "knowledge_explain"],
    ],
    domain="explanation",
    difficulty="complex_chain",
    rationale="Need base procedure (search), recent changes (temporal), then synthesize explanation.",
)

SCENARIO_15 = ToolUseScenario(
    scenario_id="TU15",
    task_description="Store the fact that the new API rate limit is 1000 requests per minute, verify it matches the documentation, and alert the team.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["fact_store", "memory_search", "fact_verify", "alert"],
    incorrect_alternatives=[
        ["fact_store", "alert"],
        ["fact_store", "fact_verify", "alert"],
        ["memory_search", "fact_store", "alert"],
    ],
    domain="fact_storage",
    difficulty="complex_chain",
    rationale="Store new limit, search for documentation, verify consistency, then alert team.",
)

SCENARIO_16 = ToolUseScenario(
    scenario_id="TU16",
    task_description="How has the team's velocity changed over the last 3 sprints?",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["temporal_query", "compare"],
    incorrect_alternatives=[
        ["memory_search", "compare"],
        ["temporal_query", "summarize"],
        ["compare"],
    ],
    domain="memory_search",
    difficulty="chain",
    rationale="Time-bounded data retrieval then comparison across the sprints.",
)

SCENARIO_17 = ToolUseScenario(
    scenario_id="TU17",
    task_description="Classify the incoming support ticket about slow page loads and check if similar issues have been reported.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["classify", "memory_search"],
    incorrect_alternatives=[
        ["memory_search"],
        ["classify"],
        ["memory_search", "classify"],
    ],
    domain="verification",
    difficulty="chain",
    rationale="Classify the ticket type first for proper routing, then search for similar reports.",
)

SCENARIO_18 = ToolUseScenario(
    scenario_id="TU18",
    task_description="Create a summary of all known security incidents from the past year for the board presentation.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["temporal_query", "classify", "summarize"],
    incorrect_alternatives=[
        ["memory_search", "summarize"],
        ["temporal_query", "summarize"],
        ["memory_search", "classify", "summarize"],
    ],
    domain="explanation",
    difficulty="complex_chain",
    rationale="Retrieve incidents by time range, classify by severity, then produce executive summary.",
)

SCENARIO_19 = ToolUseScenario(
    scenario_id="TU19",
    task_description="Find all facts related to the Kubernetes cluster, check for any gaps in our monitoring coverage, and explain the current state.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["memory_search", "gap_finder", "knowledge_explain"],
    incorrect_alternatives=[
        ["memory_search", "knowledge_explain"],
        ["gap_finder", "knowledge_explain"],
        ["memory_search", "gap_finder", "summarize"],
    ],
    domain="gap_finding",
    difficulty="complex_chain",
    rationale="Retrieve cluster facts, find monitoring gaps, then explain the full picture including gaps.",
)

SCENARIO_20 = ToolUseScenario(
    scenario_id="TU20",
    task_description="Verify that all three data sources agree on the total revenue figure for Q2.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["memory_search", "fact_verify"],
    incorrect_alternatives=[
        ["memory_search", "compare"],
        ["fact_verify"],
        ["memory_search", "compare", "fact_verify"],
    ],
    domain="verification",
    difficulty="chain",
    rationale="Retrieve all three revenue figures, then cross-reference for consistency.",
)

SCENARIO_21 = ToolUseScenario(
    scenario_id="TU21",
    task_description="An engineer reports that the staging environment is down. Log the incident, check for related infrastructure issues, and alert the on-call team.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["fact_store", "memory_search", "classify", "alert"],
    incorrect_alternatives=[
        ["alert"],
        ["fact_store", "alert"],
        ["memory_search", "alert"],
    ],
    domain="verification",
    difficulty="complex_chain",
    rationale="Log the incident, search for related issues, classify impact, then alert the team.",
)

SCENARIO_22 = ToolUseScenario(
    scenario_id="TU22",
    task_description="Explain the difference between our current and previous authentication mechanisms.",
    available_tools=ALL_TOOLS,
    expected_tool_sequence=["memory_search", "compare", "knowledge_explain"],
    incorrect_alternatives=[
        ["memory_search", "knowledge_explain"],
        ["compare", "knowledge_explain"],
        ["memory_search", "compare"],
    ],
    domain="explanation",
    difficulty="complex_chain",
    rationale="Retrieve both mechanisms, compare them, then generate a clear explanation.",
)


ALL_TOOL_USE_SCENARIOS = [
    SCENARIO_01, SCENARIO_02, SCENARIO_03, SCENARIO_04, SCENARIO_05,
    SCENARIO_06, SCENARIO_07, SCENARIO_08, SCENARIO_09, SCENARIO_10,
    SCENARIO_11, SCENARIO_12, SCENARIO_13, SCENARIO_14, SCENARIO_15,
    SCENARIO_16, SCENARIO_17, SCENARIO_18, SCENARIO_19, SCENARIO_20,
    SCENARIO_21, SCENARIO_22,
]


def get_scenario_by_id(scenario_id: str) -> ToolUseScenario | None:
    """Get a tool use scenario by its ID."""
    for s in ALL_TOOL_USE_SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    return None


def get_scenarios_by_domain(domain: str) -> list[ToolUseScenario]:
    """Get all scenarios for a given domain."""
    return [s for s in ALL_TOOL_USE_SCENARIOS if s.domain == domain]


def get_scenarios_by_difficulty(difficulty: str) -> list[ToolUseScenario]:
    """Get all scenarios for a given difficulty level."""
    return [s for s in ALL_TOOL_USE_SCENARIOS if s.difficulty == difficulty]


__all__ = [
    "ToolDefinition",
    "ToolUseScenario",
    "ALL_TOOLS",
    "ALL_TOOL_USE_SCENARIOS",
    "get_scenario_by_id",
    "get_scenarios_by_domain",
    "get_scenarios_by_difficulty",
]
