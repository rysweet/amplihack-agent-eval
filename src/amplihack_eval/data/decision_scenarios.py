"""Decision-from-memory scenarios for L16 evaluation.

Questions that require the agent to RECALL facts, ANALYZE them, and then make
a DECISION. The agent must demonstrate the full chain: recall -> reason -> decide.

Domains: security, project management, infrastructure, hiring, resource allocation.

Philosophy: Tests the highest cognitive level - using stored knowledge to make
decisions, not just retrieving or explaining it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContextFact:
    """A fact that the agent should have learned before the decision question."""

    content: str
    relevance: str  # Why this fact matters for the decision


@dataclass
class DecisionScenario:
    """A scenario requiring a decision based on recalled facts."""

    scenario_id: str
    domain: str  # security, project_management, infrastructure, hiring, resource_allocation
    context_facts: list[ContextFact]  # Facts the agent should know
    decision_question: str
    expected_decision: str  # The correct decision
    required_facts_for_decision: list[str]  # Which facts MUST be referenced
    reasoning_chain: str  # Expected chain of reasoning
    alternative_acceptable_decisions: list[str]  # Other defensible decisions
    difficulty: str  # moderate, hard, very_hard


# ── Security domain ──────────────────────────────────────────────────

SCENARIO_D01 = DecisionScenario(
    scenario_id="D01",
    domain="security",
    context_facts=[
        ContextFact(
            content="A brute force attack from 192.168.1.45 was detected on February 10, 2026.",
            relevance="The attack source IP",
        ),
        ContextFact(
            content="The current firewall allows all traffic from the 192.168.1.0/24 subnet for internal services.",
            relevance="Current firewall rule that permits the attacker's IP",
        ),
        ContextFact(
            content="The brute force attack targeted the SSH service on port 22.",
            relevance="The specific service targeted",
        ),
    ],
    decision_question="Given the brute force attack from 192.168.1.45, which firewall rule should be modified and how?",
    expected_decision=(
        "Block 192.168.1.45 specifically on port 22 (SSH) while keeping the broader "
        "192.168.1.0/24 subnet access for other internal services. Add rate limiting "
        "for SSH connections from the subnet."
    ),
    required_facts_for_decision=[
        "attack from 192.168.1.45",
        "firewall allows 192.168.1.0/24",
        "SSH on port 22",
    ],
    reasoning_chain=(
        "1. Attack came from 192.168.1.45 targeting SSH. "
        "2. Current rules allow entire 192.168.1.0/24 subnet. "
        "3. Blocking the entire subnet would disrupt internal services. "
        "4. Therefore, block only the specific IP on port 22 and add rate limiting."
    ),
    alternative_acceptable_decisions=[
        "Block 192.168.1.45 entirely while monitoring the subnet",
        "Add SSH rate limiting for the entire 192.168.1.0/24 subnet",
    ],
    difficulty="moderate",
)

SCENARIO_D02 = DecisionScenario(
    scenario_id="D02",
    domain="security",
    context_facts=[
        ContextFact(
            content="Three failed login attempts from different countries (Brazil, Russia, Nigeria) occurred within 5 minutes on the admin account.",
            relevance="Pattern suggesting credential stuffing",
        ),
        ContextFact(
            content="The admin account does not have multi-factor authentication enabled.",
            relevance="Current vulnerability",
        ),
        ContextFact(
            content="The password policy requires 12+ characters with complexity requirements.",
            relevance="Existing protection measure",
        ),
    ],
    decision_question="What immediate security actions should be taken for the admin account?",
    expected_decision=(
        "1. Immediately enable MFA on the admin account. "
        "2. Force a password reset. "
        "3. Temporarily restrict admin access to known IP ranges. "
        "4. Review the account's recent activity for unauthorized access."
    ),
    required_facts_for_decision=[
        "failed logins from multiple countries",
        "no MFA enabled",
        "password policy exists",
    ],
    reasoning_chain=(
        "1. Multi-country login attempts = credential stuffing attack. "
        "2. No MFA means password is the only protection. "
        "3. Password policy exists but may be insufficient against stolen credentials. "
        "4. Therefore, enable MFA immediately and reset password."
    ),
    alternative_acceptable_decisions=[
        "Lock the admin account temporarily and enable MFA before re-enabling",
        "Enable MFA and implement geo-blocking for admin access",
    ],
    difficulty="moderate",
)

SCENARIO_D03 = DecisionScenario(
    scenario_id="D03",
    domain="security",
    context_facts=[
        ContextFact(
            content="CVE-2026-1234 is a critical remote code execution vulnerability in the logging library log4next v2.14.",
            relevance="The vulnerability details",
        ),
        ContextFact(
            content="Our production systems use log4next v2.14.1.",
            relevance="We are running a potentially affected version",
        ),
        ContextFact(
            content="The patch for CVE-2026-1234 is available in log4next v2.14.2.",
            relevance="Fix is available",
        ),
        ContextFact(
            content="Our next maintenance window is scheduled for Saturday at 2:00 AM UTC.",
            relevance="When we can normally deploy",
        ),
    ],
    decision_question="Should we wait for the maintenance window to patch CVE-2026-1234 or deploy an emergency patch?",
    expected_decision=(
        "Deploy an emergency patch immediately. A critical RCE vulnerability in a "
        "library we actively use (v2.14.1 is affected, fix in v2.14.2) cannot wait "
        "for the next maintenance window. The risk of exploitation outweighs the "
        "risk of off-schedule deployment."
    ),
    required_facts_for_decision=[
        "critical RCE vulnerability",
        "we use affected version 2.14.1",
        "patch available in 2.14.2",
        "maintenance window timing",
    ],
    reasoning_chain=(
        "1. CVE-2026-1234 is critical severity (RCE). "
        "2. Our v2.14.1 is affected. "
        "3. Patch exists (v2.14.2) - no development needed. "
        "4. Waiting for Saturday exposes us to exploitation. "
        "5. Emergency patch is warranted."
    ),
    alternative_acceptable_decisions=[
        "Deploy emergency patch with WAF rules as interim mitigation",
    ],
    difficulty="hard",
)

# ── Project management domain ────────────────────────────────────────

SCENARIO_D04 = DecisionScenario(
    scenario_id="D04",
    domain="project_management",
    context_facts=[
        ContextFact(
            content="Project Atlas has a deadline of September 1, 2025.",
            relevance="The target date",
        ),
        ContextFact(
            content="The team completes an average of 45 story points per sprint (2-week sprints).",
            relevance="Team velocity",
        ),
        ContextFact(
            content="There are 180 story points remaining in the Project Atlas backlog.",
            relevance="Remaining work",
        ),
        ContextFact(
            content="Today is June 1, 2025.",
            relevance="Current date for calculation",
        ),
    ],
    decision_question="Based on Project Atlas's current deadline and the team's velocity, will they finish on time?",
    expected_decision=(
        "The team will likely finish on time but with very little margin. "
        "180 remaining points / 45 points per sprint = 4 sprints needed. "
        "4 sprints x 2 weeks = 8 weeks. June 1 + 8 weeks = ~July 27. "
        "This gives about 5 weeks of buffer before the September 1 deadline, "
        "which is sufficient but should be monitored closely for scope creep."
    ),
    required_facts_for_decision=[
        "September 1 deadline",
        "45 points per sprint",
        "180 points remaining",
        "current date June 1",
    ],
    reasoning_chain=(
        "1. 180 points / 45 per sprint = 4 sprints. "
        "2. 4 sprints x 2 weeks = 8 weeks. "
        "3. June 1 + 8 weeks = late July. "
        "4. September 1 deadline gives ~5 weeks buffer. "
        "5. On track but should monitor."
    ),
    alternative_acceptable_decisions=[
        "Likely on track with 5-week buffer, but recommend no new scope additions",
        "Should finish by end of July, well before the September 1 deadline",
    ],
    difficulty="moderate",
)

SCENARIO_D05 = DecisionScenario(
    scenario_id="D05",
    domain="project_management",
    context_facts=[
        ContextFact(
            content="Project Beacon is co-led by Elena Rodriguez and David Kim.",
            relevance="Current leadership",
        ),
        ContextFact(
            content="Elena Rodriguez is going on a 3-month sabbatical starting next week.",
            relevance="Upcoming leadership gap",
        ),
        ContextFact(
            content="Project Beacon has a critical demo to stakeholders in 6 weeks.",
            relevance="Upcoming milestone",
        ),
        ContextFact(
            content="David Kim has been on the project for only 2 months.",
            relevance="Experience level of remaining lead",
        ),
    ],
    decision_question="What should be done about Project Beacon's leadership before Elena's sabbatical?",
    expected_decision=(
        "Appoint a temporary co-lead or senior advisor to support David Kim during "
        "Elena's absence, especially for the stakeholder demo in 6 weeks. David has "
        "only 2 months of experience on the project. Options: promote a senior team "
        "member or bring in a PM from another team to ensure continuity."
    ),
    required_facts_for_decision=[
        "Elena leaving for sabbatical",
        "critical demo in 6 weeks",
        "David has only 2 months experience",
    ],
    reasoning_chain=(
        "1. Elena is one of two co-leads and is leaving. "
        "2. David has been on project only 2 months - limited institutional knowledge. "
        "3. Critical demo in 6 weeks requires strong leadership. "
        "4. David alone may struggle with both technical and stakeholder management. "
        "5. Need to supplement David's leadership for the demo period."
    ),
    alternative_acceptable_decisions=[
        "Have Elena do thorough knowledge transfer before leaving and assign a mentor to David",
        "Delay the demo by 2 weeks and have David lead with extra support",
    ],
    difficulty="hard",
)

SCENARIO_D06 = DecisionScenario(
    scenario_id="D06",
    domain="project_management",
    context_facts=[
        ContextFact(
            content="Mobile App v3 targets iOS, Android, and Web PWA.",
            relevance="Current scope",
        ),
        ContextFact(
            content="The development team has 8 engineers: 3 iOS, 3 Android, 2 web.",
            relevance="Resource distribution",
        ),
        ContextFact(
            content="iOS development is 80% complete, Android is 60% complete, Web PWA is 30% complete.",
            relevance="Current progress",
        ),
        ContextFact(
            content="The launch date is fixed and cannot move.",
            relevance="Hard constraint",
        ),
    ],
    decision_question="Given the current progress and fixed launch date, how should resources be reallocated?",
    expected_decision=(
        "Move 1 iOS engineer to help with Web PWA (iOS is 80% done, least at risk). "
        "Keep Android team intact (60% needs focused effort). "
        "Web PWA at 30% is the biggest risk and needs the most help. "
        "If possible, bring in 1 additional web developer."
    ),
    required_facts_for_decision=[
        "iOS 80%, Android 60%, Web 30%",
        "3 iOS, 3 Android, 2 web engineers",
        "fixed launch date",
    ],
    reasoning_chain=(
        "1. Web PWA at 30% is furthest behind with only 2 engineers. "
        "2. iOS at 80% is closest to done and can spare resources. "
        "3. Android at 60% needs its full team. "
        "4. Fixed date means we can't extend - must reallocate. "
        "5. Move iOS resources to Web PWA."
    ),
    alternative_acceptable_decisions=[
        "Move 1 iOS and 1 Android engineer to Web PWA, accepting some Android risk",
        "Descope Web PWA features to match available capacity",
    ],
    difficulty="hard",
)

# ── Infrastructure domain ────────────────────────────────────────────

SCENARIO_D07 = DecisionScenario(
    scenario_id="D07",
    domain="infrastructure",
    context_facts=[
        ContextFact(
            content="Production database max_connections is set to 500.",
            relevance="Current limit",
        ),
        ContextFact(
            content="Current connection usage averages 420 connections during peak hours.",
            relevance="Current utilization",
        ),
        ContextFact(
            content="Traffic is expected to grow 30% in the next quarter due to a marketing campaign.",
            relevance="Growth projection",
        ),
        ContextFact(
            content="The database server has 64GB of RAM, each connection uses approximately 10MB.",
            relevance="Resource constraints",
        ),
    ],
    decision_question="Should the max_connections setting be increased, and if so, to what value?",
    expected_decision=(
        "Yes, increase max_connections. Current peak (420) is already at 84% of limit (500). "
        "With 30% growth: 420 * 1.3 = 546 connections needed, exceeding current limit. "
        "With 64GB RAM and ~10MB per connection: max theoretical = ~6400 connections, "
        "so memory is not the bottleneck. Recommend setting to 750 (546 * 1.37 safety margin)."
    ),
    required_facts_for_decision=[
        "max_connections 500",
        "peak usage 420",
        "30% growth expected",
        "64GB RAM, 10MB per connection",
    ],
    reasoning_chain=(
        "1. Current utilization: 420/500 = 84%. "
        "2. With 30% growth: 420 * 1.3 = 546 > 500 limit. "
        "3. Memory check: 750 * 10MB = 7.5GB, well within 64GB. "
        "4. Need to increase before the campaign launches."
    ),
    alternative_acceptable_decisions=[
        "Increase to 700 with connection pooling to optimize usage",
        "Increase to 650 and implement auto-scaling",
    ],
    difficulty="moderate",
)

SCENARIO_D08 = DecisionScenario(
    scenario_id="D08",
    domain="infrastructure",
    context_facts=[
        ContextFact(
            content="API Gateway rate limit is 1000 requests per minute per client.",
            relevance="Current rate limit",
        ),
        ContextFact(
            content="The largest client sends an average of 800 requests per minute during business hours.",
            relevance="Top client usage",
        ),
        ContextFact(
            content="The largest client has complained about rate limiting during their peak batch processing at month-end.",
            relevance="Client impact",
        ),
        ContextFact(
            content="There are 50 active API clients. The average client sends 200 requests per minute.",
            relevance="Overall usage context",
        ),
    ],
    decision_question="Should the API rate limit be increased for the largest client?",
    expected_decision=(
        "Implement a tiered rate limit rather than a blanket increase. "
        "Give the largest client an Enterprise tier with 2000 req/min to handle "
        "month-end batch processing (their peak exceeds 1000). "
        "Keep the standard tier at 1000 req/min for other clients. "
        "The average client uses only 200 req/min so the standard limit is fine for them."
    ),
    required_facts_for_decision=[
        "current limit 1000 req/min",
        "largest client at 800 average",
        "complaints about month-end peaks",
        "50 clients at 200 average",
    ],
    reasoning_chain=(
        "1. Largest client averages 800/1000 = 80% of limit. "
        "2. Month-end batch exceeds 1000, causing rate limiting. "
        "3. Blanket increase would be wasteful (average client only uses 200). "
        "4. Tiered limits solve the problem without over-provisioning."
    ),
    alternative_acceptable_decisions=[
        "Increase limit to 2000 for all clients and monitor",
        "Add burst allowance (1500 req/min for 5-minute windows) for all clients",
    ],
    difficulty="hard",
)

SCENARIO_D09 = DecisionScenario(
    scenario_id="D09",
    domain="infrastructure",
    context_facts=[
        ContextFact(
            content="Kubernetes cluster is running version 1.30.",
            relevance="Current version",
        ),
        ContextFact(
            content="Kubernetes 1.31 was released with critical security fixes but also includes breaking API changes.",
            relevance="Upgrade available with tradeoffs",
        ),
        ContextFact(
            content="Our CI pipeline tests pass on K8s 1.30 but have not been validated on 1.31.",
            relevance="Testing status",
        ),
        ContextFact(
            content="Three of our production services use the deprecated API that is removed in 1.31.",
            relevance="Breaking change impact",
        ),
    ],
    decision_question="Should we upgrade the Kubernetes cluster to 1.31?",
    expected_decision=(
        "Upgrade, but in phases. First, update the 3 services using deprecated APIs to "
        "be compatible with 1.31. Then validate CI on 1.31 in a staging environment. "
        "Finally, upgrade production. The critical security fixes make the upgrade "
        "necessary, but rushing it would break 3 production services."
    ),
    required_facts_for_decision=[
        "running K8s 1.30",
        "1.31 has security fixes and breaking changes",
        "CI not validated on 1.31",
        "3 services use deprecated API",
    ],
    reasoning_chain=(
        "1. Security fixes make upgrade important. "
        "2. Breaking API changes affect 3 services. "
        "3. CI not yet validated on 1.31. "
        "4. Phased approach: fix APIs -> validate CI -> upgrade."
    ),
    alternative_acceptable_decisions=[
        "Apply security patches to 1.30 as interim measure while preparing for 1.31",
        "Upgrade staging first, run parallel clusters during migration",
    ],
    difficulty="very_hard",
)

# ── Hiring domain ────────────────────────────────────────────────────

SCENARIO_D10 = DecisionScenario(
    scenario_id="D10",
    domain="hiring",
    context_facts=[
        ContextFact(
            content="The engineering team has 61 members.",
            relevance="Current headcount",
        ),
        ContextFact(
            content="The Security team has 6 members and handles all incident response.",
            relevance="Security team capacity",
        ),
        ContextFact(
            content="Security incidents have increased 40% quarter over quarter.",
            relevance="Growing workload",
        ),
        ContextFact(
            content="Average time to resolve security incidents has gone from 4 hours to 8 hours.",
            relevance="Declining performance metric",
        ),
    ],
    decision_question="Should we hire more security engineers? If so, how many?",
    expected_decision=(
        "Yes, hire 2-3 security engineers immediately. The team of 6 is clearly overloaded: "
        "incidents up 40% and resolution time doubled (4h to 8h). Adding 50% capacity "
        "(3 engineers) would bring the team to 9, roughly matching the 40% workload increase "
        "with some buffer for continued growth."
    ),
    required_facts_for_decision=[
        "6 person security team",
        "40% incident increase",
        "resolution time doubled",
    ],
    reasoning_chain=(
        "1. Security team of 6 handling 40% more incidents. "
        "2. Resolution time doubled = team is overwhelmed. "
        "3. 40% more work / 6 people = need ~2.4 more people minimum. "
        "4. Round up to 3 for buffer against continued growth."
    ),
    alternative_acceptable_decisions=[
        "Hire 2 senior security engineers and invest in automation tooling",
        "Hire 3 engineers and implement a security operations rotation with other teams",
    ],
    difficulty="moderate",
)

# ── Resource allocation domain ───────────────────────────────────────

SCENARIO_D11 = DecisionScenario(
    scenario_id="D11",
    domain="resource_allocation",
    context_facts=[
        ContextFact(
            content="We use Cloudflare as our CDN provider at $5,000/month.",
            relevance="Current CDN cost",
        ),
        ContextFact(
            content="Monthly bandwidth usage is 50TB and growing 10% monthly.",
            relevance="Usage trend",
        ),
        ContextFact(
            content="Cloudflare charges $0.10/GB after the first 20TB included in the plan.",
            relevance="Cost structure",
        ),
        ContextFact(
            content="AWS CloudFront offers a volume discount at $0.06/GB for commitments over 100TB/month.",
            relevance="Alternative pricing",
        ),
    ],
    decision_question="Should we switch CDN providers based on our bandwidth growth trajectory?",
    expected_decision=(
        "Not yet, but plan the switch for when bandwidth exceeds ~80TB/month. "
        "Current cost: $5,000 base + (50-20)*0.10*1000 = $5,000 + $3,000 = $8,000/month. "
        "At 10% monthly growth, we'll hit 100TB in ~7 months. "
        "At 100TB on CloudFront: 100TB * $0.06 * 1000 = $6,000/month (cheaper). "
        "But at 50TB, Cloudflare is comparable. Start planning the migration now."
    ),
    required_facts_for_decision=[
        "Cloudflare $5,000/month",
        "50TB bandwidth, 10% growth",
        "Cloudflare $0.10/GB after 20TB",
        "CloudFront $0.06/GB at 100TB+",
    ],
    reasoning_chain=(
        "1. Current: $5,000 + 30TB * $100/TB = $8,000/month. "
        "2. CloudFront at 50TB: 50TB * $60/TB = $3,000/month (cheaper, but need commitment). "
        "3. At 10% growth: ~7 months to reach 100TB. "
        "4. Plan migration for when volume discount makes economic sense."
    ),
    alternative_acceptable_decisions=[
        "Switch to CloudFront now to lock in volume pricing ahead of growth",
        "Stay with Cloudflare and negotiate a volume discount based on growth projections",
    ],
    difficulty="very_hard",
)

SCENARIO_D12 = DecisionScenario(
    scenario_id="D12",
    domain="resource_allocation",
    context_facts=[
        ContextFact(
            content="The Data Pipeline project was just completed and deployed to production.",
            relevance="Resources now available",
        ),
        ContextFact(
            content="The Data Pipeline team had 5 engineers.",
            relevance="Available engineers",
        ),
        ContextFact(
            content="Project Atlas is the highest priority and needs 3 more engineers to meet its September deadline.",
            relevance="Staffing need for priority project",
        ),
        ContextFact(
            content="The Mobile App v3 Web PWA component is significantly behind schedule.",
            relevance="Another project in trouble",
        ),
    ],
    decision_question="How should the 5 newly available Data Pipeline engineers be allocated?",
    expected_decision=(
        "Allocate 3 engineers to Project Atlas (highest priority, explicit need) and "
        "2 engineers to Mobile App v3 Web PWA (significantly behind schedule). "
        "Project Atlas gets priority as it is the highest priority project. "
        "The remaining 2 help address the Web PWA risk."
    ),
    required_facts_for_decision=[
        "5 engineers available from Data Pipeline",
        "Atlas needs 3 more engineers",
        "Atlas is highest priority",
        "Web PWA is behind schedule",
    ],
    reasoning_chain=(
        "1. Five engineers freed up from completed project. "
        "2. Atlas is highest priority and explicitly needs 3. "
        "3. Web PWA is significantly behind and needs help. "
        "4. 3 to Atlas + 2 to Web PWA = 5 total, exact match."
    ),
    alternative_acceptable_decisions=[
        "All 5 to Atlas since it's highest priority, address Web PWA separately",
        "3 to Atlas, 1 to Web PWA, 1 to technical debt/on-call rotation",
    ],
    difficulty="moderate",
)

SCENARIO_D13 = DecisionScenario(
    scenario_id="D13",
    domain="security",
    context_facts=[
        ContextFact(
            content="We use Cloudflare for CDN and DDoS protection.",
            relevance="Current protection",
        ),
        ContextFact(
            content="Our API rate limit is 1000 req/min per client.",
            relevance="Application-layer protection",
        ),
        ContextFact(
            content="A competitor was hit by a 500Gbps DDoS attack last week and their service was down for 6 hours.",
            relevance="Threat intelligence",
        ),
        ContextFact(
            content="Our current Cloudflare plan includes DDoS protection up to 100Gbps.",
            relevance="Current protection limit",
        ),
    ],
    decision_question="Given the competitor's DDoS attack, should we upgrade our DDoS protection?",
    expected_decision=(
        "Yes, upgrade the Cloudflare plan to cover higher DDoS volumes. "
        "Our current 100Gbps limit is well below the 500Gbps attack seen against "
        "our competitor. As a similar target, we should assume we could face similar "
        "attacks. Upgrade to an enterprise plan with unlimited/higher DDoS protection."
    ),
    required_facts_for_decision=[
        "Cloudflare for DDoS protection",
        "competitor hit with 500Gbps",
        "our limit is 100Gbps",
    ],
    reasoning_chain=(
        "1. Competitor attacked at 500Gbps. "
        "2. Our protection only covers 100Gbps. "
        "3. Similar industry = similar threat profile. "
        "4. Must upgrade before we become the next target."
    ),
    alternative_acceptable_decisions=[
        "Upgrade Cloudflare and add a secondary DDoS mitigation provider for redundancy",
        "Keep current plan but add application-layer protections and incident response plan",
    ],
    difficulty="moderate",
)

SCENARIO_D14 = DecisionScenario(
    scenario_id="D14",
    domain="infrastructure",
    context_facts=[
        ContextFact(
            content="CPU alert threshold is set to 75% for production servers.",
            relevance="Current monitoring config",
        ),
        ContextFact(
            content="Average CPU usage during peak is 65% across production servers.",
            relevance="Current utilization",
        ),
        ContextFact(
            content="Last month there were 47 false-positive CPU alerts during normal batch processing.",
            relevance="Alert fatigue issue",
        ),
        ContextFact(
            content="Batch processing jobs run nightly and regularly spike CPU to 72-78% for 30 minutes.",
            relevance="Known spike pattern",
        ),
    ],
    decision_question="How should the CPU alert threshold be adjusted to reduce false positives without missing real issues?",
    expected_decision=(
        "Implement a two-tier alert system: raise the static threshold to 85% to avoid "
        "batch processing false positives (spikes go to 72-78%), and add a duration-based "
        "alert that triggers at 75% sustained for more than 45 minutes (batch jobs only "
        "run for 30 min). This eliminates the 47 false positives from batch jobs while "
        "catching genuine sustained high CPU."
    ),
    required_facts_for_decision=[
        "current threshold 75%",
        "batch processing spikes to 72-78%",
        "47 false positive alerts",
        "batch runs 30 minutes",
    ],
    reasoning_chain=(
        "1. Batch jobs spike to 72-78%, triggering the 75% threshold. "
        "2. This is normal behavior lasting only 30 minutes. "
        "3. Simple threshold raise to 85% would miss sustained 80% CPU issues. "
        "4. Duration-based alerting (75% for >45min) filters batch but catches real problems."
    ),
    alternative_acceptable_decisions=[
        "Raise threshold to 80% as a simple fix (catches most batch spikes but not all)",
        "Implement time-of-day based thresholds (higher during batch window)",
    ],
    difficulty="very_hard",
)

SCENARIO_D15 = DecisionScenario(
    scenario_id="D15",
    domain="project_management",
    context_facts=[
        ContextFact(
            content="The engineering team has 61 members across 8 teams.",
            relevance="Organization size",
        ),
        ContextFact(
            content="Three teams are working on Project Atlas, two on Mobile App v3, one on infrastructure, one on security, one on platform.",
            relevance="Team allocation",
        ),
        ContextFact(
            content="The security team is understaffed (6 members, incidents up 40%).",
            relevance="Known pain point",
        ),
        ContextFact(
            content="Project Atlas is 70% complete and ahead of schedule by 2 weeks.",
            relevance="Project status - doing well",
        ),
        ContextFact(
            content="Mobile App v3 Web PWA is only 30% complete.",
            relevance="Project status - behind",
        ),
    ],
    decision_question="A major client requests a custom integration that requires 4 engineers for 3 months. Where should the engineers come from?",
    expected_decision=(
        "Pull 2 engineers from one of the three Atlas teams (Atlas is ahead of schedule "
        "and can absorb the loss) and 2 from the platform team. Do NOT pull from: "
        "Security (already understaffed), Mobile App v3 (behind schedule), or "
        "Infrastructure (typically critical). Monitor Atlas progress after the reduction."
    ),
    required_facts_for_decision=[
        "Atlas has 3 teams and is ahead of schedule",
        "security is understaffed",
        "Web PWA is behind",
        "need 4 engineers",
    ],
    reasoning_chain=(
        "1. Need 4 engineers from existing teams. "
        "2. Security is out - already understaffed with growing incidents. "
        "3. Mobile App v3 is out - Web PWA is critically behind. "
        "4. Infrastructure is out - too critical to reduce. "
        "5. Atlas has 3 teams AND is ahead of schedule - can spare 2. "
        "6. Platform can contribute 2. "
        "7. Total: 4 engineers without impacting at-risk projects."
    ),
    alternative_acceptable_decisions=[
        "Pull all 4 from Atlas teams since they are ahead of schedule",
        "Pull 2 from Atlas, hire 2 contractors for the integration",
    ],
    difficulty="very_hard",
)


ALL_DECISION_SCENARIOS = [
    SCENARIO_D01, SCENARIO_D02, SCENARIO_D03, SCENARIO_D04, SCENARIO_D05,
    SCENARIO_D06, SCENARIO_D07, SCENARIO_D08, SCENARIO_D09, SCENARIO_D10,
    SCENARIO_D11, SCENARIO_D12, SCENARIO_D13, SCENARIO_D14, SCENARIO_D15,
]


def get_decision_scenario_by_id(scenario_id: str) -> DecisionScenario | None:
    """Get a decision scenario by its ID."""
    for s in ALL_DECISION_SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    return None


def get_decision_scenarios_by_domain(domain: str) -> list[DecisionScenario]:
    """Get all decision scenarios for a given domain."""
    return [s for s in ALL_DECISION_SCENARIOS if s.domain == domain]


def get_decision_scenarios_by_difficulty(difficulty: str) -> list[DecisionScenario]:
    """Get all decision scenarios for a given difficulty."""
    return [s for s in ALL_DECISION_SCENARIOS if s.difficulty == difficulty]


__all__ = [
    "ContextFact",
    "DecisionScenario",
    "ALL_DECISION_SCENARIOS",
    "get_decision_scenario_by_id",
    "get_decision_scenarios_by_domain",
    "get_decision_scenarios_by_difficulty",
]
