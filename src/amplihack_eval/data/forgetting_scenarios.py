"""Selective forgetting scenarios for L14 evaluation.

Generates dialogues where facts get SUPERSEDED over time. The agent must return
the CURRENT value and not be confused by old, superseded values.

Domains: people (role changes), projects (deadline/budget updates),
infrastructure (config changes).

Philosophy: Each scenario is a self-contained timeline of updates. The agent
should track the latest value and correctly note that earlier values were
superseded.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FactUpdate:
    """A single update to a fact, representing one point in time."""

    timestamp: str  # ISO format
    content: str  # The statement at this point in time
    value: str  # The specific value being tracked
    is_current: bool = False  # Whether this is the final/current value


@dataclass
class ForgettingScenario:
    """A scenario where facts get superseded over time."""

    scenario_id: str
    domain: str  # people, projects, infrastructure
    entity: str  # The thing being tracked (person, project, server, etc.)
    attribute: str  # What changes (role, deadline, config, etc.)
    updates: list[FactUpdate]  # Chronological list of updates
    question: str  # The question to ask
    expected_current_value: str  # What the agent SHOULD answer
    superseded_values: list[str]  # Values that should NOT appear as current
    rationale: str


# ── People domain: Role changes ─────────────────────────────────────

SCENARIO_F01 = ForgettingScenario(
    scenario_id="F01",
    domain="people",
    entity="Sarah Chen",
    attribute="role",
    updates=[
        FactUpdate(
            timestamp="2025-01-15T09:00:00Z",
            content="Sarah Chen joined the company as a Junior Engineer.",
            value="Junior Engineer",
        ),
        FactUpdate(
            timestamp="2025-06-01T09:00:00Z",
            content="Sarah Chen has been promoted to Senior Engineer after strong Q1 and Q2 performance.",
            value="Senior Engineer",
        ),
        FactUpdate(
            timestamp="2026-01-10T09:00:00Z",
            content="Sarah Chen has been promoted to Engineering Director, effective immediately.",
            value="Engineering Director",
            is_current=True,
        ),
    ],
    question="What is Sarah Chen's current role?",
    expected_current_value="Engineering Director",
    superseded_values=["Junior Engineer", "Senior Engineer"],
    rationale="Agent must return the most recent role, not earlier positions.",
)

SCENARIO_F02 = ForgettingScenario(
    scenario_id="F02",
    domain="people",
    entity="Marcus Johnson",
    attribute="team",
    updates=[
        FactUpdate(
            timestamp="2025-03-01T09:00:00Z",
            content="Marcus Johnson is assigned to the Frontend team.",
            value="Frontend team",
        ),
        FactUpdate(
            timestamp="2025-08-15T09:00:00Z",
            content="Marcus Johnson has transferred to the Platform team to work on infrastructure.",
            value="Platform team",
        ),
        FactUpdate(
            timestamp="2026-02-01T09:00:00Z",
            content="Marcus Johnson has moved to the Security team as part of the org restructuring.",
            value="Security team",
            is_current=True,
        ),
    ],
    question="Which team is Marcus Johnson on?",
    expected_current_value="Security team",
    superseded_values=["Frontend team", "Platform team"],
    rationale="Agent must track the latest team assignment.",
)

SCENARIO_F03 = ForgettingScenario(
    scenario_id="F03",
    domain="people",
    entity="Dr. Aisha Patel",
    attribute="office_location",
    updates=[
        FactUpdate(
            timestamp="2025-01-01T09:00:00Z",
            content="Dr. Aisha Patel works from the Seattle office, Building A, Floor 3.",
            value="Seattle office, Building A, Floor 3",
        ),
        FactUpdate(
            timestamp="2025-07-01T09:00:00Z",
            content="Dr. Aisha Patel has relocated to the Austin office as part of the team consolidation.",
            value="Austin office",
        ),
        FactUpdate(
            timestamp="2026-01-15T09:00:00Z",
            content="Dr. Aisha Patel is now working remotely from Portland on a permanent basis.",
            value="Remote from Portland",
            is_current=True,
        ),
    ],
    question="Where does Dr. Aisha Patel work?",
    expected_current_value="Remote from Portland",
    superseded_values=["Seattle office, Building A, Floor 3", "Austin office"],
    rationale="Agent must return current work location, not prior offices.",
)

# ── Projects domain: Deadline and budget updates ─────────────────────

SCENARIO_F04 = ForgettingScenario(
    scenario_id="F04",
    domain="projects",
    entity="Project Atlas",
    attribute="deadline",
    updates=[
        FactUpdate(
            timestamp="2025-02-01T09:00:00Z",
            content="Project Atlas has a target delivery date of June 15, 2025.",
            value="June 15, 2025",
        ),
        FactUpdate(
            timestamp="2025-05-20T09:00:00Z",
            content="Due to scope changes, Project Atlas deadline has been pushed to August 3, 2025.",
            value="August 3, 2025",
        ),
        FactUpdate(
            timestamp="2025-07-15T09:00:00Z",
            content="Project Atlas deadline extended again to September 1, 2025 due to vendor delays.",
            value="September 1, 2025",
            is_current=True,
        ),
    ],
    question="When is the Project Atlas deadline?",
    expected_current_value="September 1, 2025",
    superseded_values=["June 15, 2025", "August 3, 2025"],
    rationale="Agent must return the latest deadline, not earlier estimates.",
)

SCENARIO_F05 = ForgettingScenario(
    scenario_id="F05",
    domain="projects",
    entity="Project Atlas",
    attribute="budget",
    updates=[
        FactUpdate(
            timestamp="2025-01-10T09:00:00Z",
            content="Project Atlas has an approved budget of $2.5 million.",
            value="$2.5 million",
        ),
        FactUpdate(
            timestamp="2025-04-01T09:00:00Z",
            content="Project Atlas budget increased to $3.1 million after adding the mobile component.",
            value="$3.1 million",
        ),
        FactUpdate(
            timestamp="2025-08-01T09:00:00Z",
            content="Final budget for Project Atlas approved at $3.8 million including contingency reserves.",
            value="$3.8 million",
            is_current=True,
        ),
    ],
    question="What is the budget for Project Atlas?",
    expected_current_value="$3.8 million",
    superseded_values=["$2.5 million", "$3.1 million"],
    rationale="Agent must return the final approved budget.",
)

SCENARIO_F06 = ForgettingScenario(
    scenario_id="F06",
    domain="projects",
    entity="Project Beacon",
    attribute="lead",
    updates=[
        FactUpdate(
            timestamp="2025-03-01T09:00:00Z",
            content="Project Beacon is led by James Wright.",
            value="James Wright",
        ),
        FactUpdate(
            timestamp="2025-09-15T09:00:00Z",
            content="Project Beacon leadership transferred to Elena Rodriguez after James Wright's departure.",
            value="Elena Rodriguez",
        ),
        FactUpdate(
            timestamp="2026-01-20T09:00:00Z",
            content="Project Beacon is now co-led by Elena Rodriguez and David Kim.",
            value="Elena Rodriguez and David Kim (co-leads)",
            is_current=True,
        ),
    ],
    question="Who leads Project Beacon?",
    expected_current_value="Elena Rodriguez and David Kim (co-leads)",
    superseded_values=["James Wright", "Elena Rodriguez"],
    rationale="Agent must know the current co-leadership, not prior single leads.",
)

SCENARIO_F07 = ForgettingScenario(
    scenario_id="F07",
    domain="projects",
    entity="Mobile App v3",
    attribute="target_platform",
    updates=[
        FactUpdate(
            timestamp="2025-02-15T09:00:00Z",
            content="Mobile App v3 will target iOS only for initial launch.",
            value="iOS only",
        ),
        FactUpdate(
            timestamp="2025-05-01T09:00:00Z",
            content="Mobile App v3 scope expanded to target both iOS and Android simultaneously.",
            value="iOS and Android",
        ),
        FactUpdate(
            timestamp="2025-10-01T09:00:00Z",
            content="Mobile App v3 will now also include a web PWA alongside iOS and Android.",
            value="iOS, Android, and Web PWA",
            is_current=True,
        ),
    ],
    question="What platforms does Mobile App v3 target?",
    expected_current_value="iOS, Android, and Web PWA",
    superseded_values=["iOS only", "iOS and Android"],
    rationale="Agent must return the full current platform set.",
)

# ── Infrastructure domain: Config changes ────────────────────────────

SCENARIO_F08 = ForgettingScenario(
    scenario_id="F08",
    domain="infrastructure",
    entity="Production Database",
    attribute="max_connections",
    updates=[
        FactUpdate(
            timestamp="2025-01-01T09:00:00Z",
            content="Production database max_connections set to 100.",
            value="100",
        ),
        FactUpdate(
            timestamp="2025-06-01T09:00:00Z",
            content="Production database max_connections increased to 250 due to traffic growth.",
            value="250",
        ),
        FactUpdate(
            timestamp="2025-12-01T09:00:00Z",
            content="Production database max_connections raised to 500 after hardware upgrade.",
            value="500",
            is_current=True,
        ),
    ],
    question="What is the max_connections setting for the production database?",
    expected_current_value="500",
    superseded_values=["100", "250"],
    rationale="Agent must return the current config value, not earlier settings.",
)

SCENARIO_F09 = ForgettingScenario(
    scenario_id="F09",
    domain="infrastructure",
    entity="API Gateway",
    attribute="rate_limit",
    updates=[
        FactUpdate(
            timestamp="2025-03-01T09:00:00Z",
            content="API Gateway rate limit set to 100 requests per minute per client.",
            value="100 req/min",
        ),
        FactUpdate(
            timestamp="2025-07-01T09:00:00Z",
            content="API Gateway rate limit increased to 500 requests per minute per client.",
            value="500 req/min",
        ),
        FactUpdate(
            timestamp="2025-11-15T09:00:00Z",
            content="API Gateway rate limit finalized at 1000 requests per minute per client after load testing.",
            value="1000 req/min",
            is_current=True,
        ),
    ],
    question="What is the current API Gateway rate limit?",
    expected_current_value="1000 requests per minute per client",
    superseded_values=["100 req/min", "500 req/min"],
    rationale="Agent must return the latest rate limit configuration.",
)

SCENARIO_F10 = ForgettingScenario(
    scenario_id="F10",
    domain="infrastructure",
    entity="CDN Provider",
    attribute="provider",
    updates=[
        FactUpdate(
            timestamp="2025-01-15T09:00:00Z",
            content="We use CloudFront as our CDN provider.",
            value="CloudFront",
        ),
        FactUpdate(
            timestamp="2025-08-01T09:00:00Z",
            content="CDN migrated from CloudFront to Fastly for better edge computing support.",
            value="Fastly",
        ),
        FactUpdate(
            timestamp="2026-02-01T09:00:00Z",
            content="CDN provider switched to Cloudflare for cost optimization and integrated DDoS protection.",
            value="Cloudflare",
            is_current=True,
        ),
    ],
    question="What CDN provider do we use?",
    expected_current_value="Cloudflare",
    superseded_values=["CloudFront", "Fastly"],
    rationale="Agent must know the current CDN provider, not historical ones.",
)

SCENARIO_F11 = ForgettingScenario(
    scenario_id="F11",
    domain="infrastructure",
    entity="Kubernetes Cluster",
    attribute="version",
    updates=[
        FactUpdate(
            timestamp="2025-02-01T09:00:00Z",
            content="Production Kubernetes cluster running version 1.27.",
            value="1.27",
        ),
        FactUpdate(
            timestamp="2025-06-15T09:00:00Z",
            content="Kubernetes cluster upgraded to version 1.28 during maintenance window.",
            value="1.28",
        ),
        FactUpdate(
            timestamp="2025-11-01T09:00:00Z",
            content="Kubernetes cluster upgraded to version 1.29 with enhanced security features.",
            value="1.29",
        ),
        FactUpdate(
            timestamp="2026-02-10T09:00:00Z",
            content="Kubernetes cluster upgraded to version 1.30, the latest LTS release.",
            value="1.30",
            is_current=True,
        ),
    ],
    question="What version of Kubernetes is running in production?",
    expected_current_value="1.30",
    superseded_values=["1.27", "1.28", "1.29"],
    rationale="Agent must return the most recent Kubernetes version (4 updates deep).",
)

SCENARIO_F12 = ForgettingScenario(
    scenario_id="F12",
    domain="people",
    entity="Engineering Team",
    attribute="size",
    updates=[
        FactUpdate(
            timestamp="2025-01-01T09:00:00Z",
            content="The engineering team has 45 members.",
            value="45",
        ),
        FactUpdate(
            timestamp="2025-04-01T09:00:00Z",
            content="After Q1 hiring, the engineering team grew to 52 members.",
            value="52",
        ),
        FactUpdate(
            timestamp="2025-10-01T09:00:00Z",
            content="Following the restructuring, the engineering team is now 48 members.",
            value="48",
        ),
        FactUpdate(
            timestamp="2026-01-01T09:00:00Z",
            content="With the new year hires, the engineering team has 61 members.",
            value="61",
            is_current=True,
        ),
    ],
    question="How many members are on the engineering team?",
    expected_current_value="61",
    superseded_values=["45", "52", "48"],
    rationale="Agent must return the latest headcount, not any prior values.",
)

SCENARIO_F13 = ForgettingScenario(
    scenario_id="F13",
    domain="projects",
    entity="Data Pipeline",
    attribute="status",
    updates=[
        FactUpdate(
            timestamp="2025-05-01T09:00:00Z",
            content="Data Pipeline project status: In Planning.",
            value="In Planning",
        ),
        FactUpdate(
            timestamp="2025-08-01T09:00:00Z",
            content="Data Pipeline project has moved to Active Development.",
            value="Active Development",
        ),
        FactUpdate(
            timestamp="2025-12-15T09:00:00Z",
            content="Data Pipeline project is now in Testing phase.",
            value="Testing",
        ),
        FactUpdate(
            timestamp="2026-02-01T09:00:00Z",
            content="Data Pipeline project has been deployed to production. Status: Complete.",
            value="Complete (in production)",
            is_current=True,
        ),
    ],
    question="What is the status of the Data Pipeline project?",
    expected_current_value="Complete (in production)",
    superseded_values=["In Planning", "Active Development", "Testing"],
    rationale="Agent must return the final status, not intermediate phases.",
)

SCENARIO_F14 = ForgettingScenario(
    scenario_id="F14",
    domain="infrastructure",
    entity="Monitoring System",
    attribute="alert_threshold",
    updates=[
        FactUpdate(
            timestamp="2025-02-01T09:00:00Z",
            content="CPU alert threshold set to 90% for production servers.",
            value="90%",
        ),
        FactUpdate(
            timestamp="2025-09-01T09:00:00Z",
            content="CPU alert threshold lowered to 80% after the September outage investigation.",
            value="80%",
        ),
        FactUpdate(
            timestamp="2026-01-15T09:00:00Z",
            content="CPU alert threshold adjusted to 75% as recommended by the reliability team.",
            value="75%",
            is_current=True,
        ),
    ],
    question="What is the CPU alert threshold for production servers?",
    expected_current_value="75%",
    superseded_values=["90%", "80%"],
    rationale="Agent must return the latest threshold, not prior configurations.",
)

SCENARIO_F15 = ForgettingScenario(
    scenario_id="F15",
    domain="people",
    entity="Company",
    attribute="ceo",
    updates=[
        FactUpdate(
            timestamp="2024-01-01T09:00:00Z",
            content="The company CEO is Robert Martinez.",
            value="Robert Martinez",
        ),
        FactUpdate(
            timestamp="2025-06-01T09:00:00Z",
            content="Robert Martinez has stepped down. The board has appointed Lisa Wang as the new CEO.",
            value="Lisa Wang",
        ),
        FactUpdate(
            timestamp="2026-01-01T09:00:00Z",
            content="Lisa Wang has been succeeded by David Okonkwo as CEO after the merger.",
            value="David Okonkwo",
            is_current=True,
        ),
    ],
    question="Who is the CEO of the company?",
    expected_current_value="David Okonkwo",
    superseded_values=["Robert Martinez", "Lisa Wang"],
    rationale="Agent must return the current CEO, not prior leaders.",
)


ALL_FORGETTING_SCENARIOS = [
    SCENARIO_F01, SCENARIO_F02, SCENARIO_F03, SCENARIO_F04, SCENARIO_F05,
    SCENARIO_F06, SCENARIO_F07, SCENARIO_F08, SCENARIO_F09, SCENARIO_F10,
    SCENARIO_F11, SCENARIO_F12, SCENARIO_F13, SCENARIO_F14, SCENARIO_F15,
]


def get_forgetting_scenario_by_id(scenario_id: str) -> ForgettingScenario | None:
    """Get a forgetting scenario by its ID."""
    for s in ALL_FORGETTING_SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    return None


def get_forgetting_scenarios_by_domain(domain: str) -> list[ForgettingScenario]:
    """Get all forgetting scenarios for a given domain."""
    return [s for s in ALL_FORGETTING_SCENARIOS if s.domain == domain]


__all__ = [
    "FactUpdate",
    "ForgettingScenario",
    "ALL_FORGETTING_SCENARIOS",
    "get_forgetting_scenario_by_id",
    "get_forgetting_scenarios_by_domain",
]
