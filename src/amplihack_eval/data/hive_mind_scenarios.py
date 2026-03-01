"""Hive mind evaluation scenarios for multi-agent knowledge sharing.

Tests whether a group of agents connected via a shared memory ("hive mind")
can answer questions that require knowledge from multiple agents' domains.

Scenarios:
1. Infrastructure Team: networking, storage, compute, security, monitoring
2. Software Architecture: frontend, backend, database, devops, testing
3. Incident Response: timeline, logs, metrics, code changes, communications
4. Research Synthesis: 5 agents each read different papers on the same topic
5. Adversarial Resilience: 4 correct agents + 1 agent with misleading facts

Each scenario has 20 facts per agent and 15 questions (5 single-domain,
5 cross-domain, 5 synthesis).

Philosophy:
- Deterministic: no LLM needed for data generation
- Ground truth tracked for every fact delivered to every agent
- Scenarios test distinct failure modes of shared memory systems
- Agent-agnostic: works with any AgentAdapter-based agents

Public API:
    HiveMindQuestion: A question requiring cross-agent knowledge
    HiveMindScenario: A complete evaluation scenario
    ALL_HIVE_MIND_SCENARIOS: All 5 predefined scenarios
    get_scenario_by_id: Lookup by scenario_id
    get_scenarios_by_difficulty: Filter by question difficulty
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HiveMindQuestion:
    """A question that may require knowledge from multiple agents.

    Attributes:
        question_id: Unique identifier for the question
        text: The question text
        required_domains: Which agents' knowledge is needed to answer
        expected_keywords: Keywords that should appear in a correct answer
        difficulty: single_domain | cross_domain | synthesis
    """

    question_id: str
    text: str
    required_domains: list[str]
    expected_keywords: list[str]
    difficulty: str  # "single_domain" | "cross_domain" | "synthesis"


@dataclass
class HiveMindScenario:
    """A complete hive mind evaluation scenario.

    Attributes:
        scenario_id: Unique identifier for the scenario
        description: Human-readable description
        num_agents: Number of agents in the hive
        agent_domains: agent_id -> list of fact strings
        questions: Questions requiring cross-agent knowledge
    """

    scenario_id: str
    description: str
    num_agents: int
    agent_domains: dict[str, list[str]]
    questions: list[HiveMindQuestion]


# ============================================================
# Scenario 1: Infrastructure Team
# ============================================================

_INFRA_NETWORKING_FACTS = [
    "The primary load balancer is an F5 BIG-IP running firmware v17.1.1.",
    "DNS resolution is handled by Route 53 with a 60-second TTL on A records.",
    "The internal network uses the 10.0.0.0/8 CIDR range with /24 subnets per team.",
    "East-west traffic between services uses mTLS enforced by Istio sidecar proxies.",
    "The CDN edge nodes are deployed in 14 global PoPs including Tokyo, Frankfurt, and Sao Paulo.",
    "Maximum bandwidth allocation per microservice is 500 Mbps with burst to 1 Gbps.",
    "VPN access uses WireGuard with certificate rotation every 90 days.",
    "The core switch fabric has 40GbE uplinks with LACP bonding.",
    "Network segmentation isolates PCI-scoped systems in VLAN 200.",
    "BGP peering with transit providers uses AS 64512 with community tagging.",
    "Latency SLA between availability zones is under 2 ms at p99.",
    "The firewall ruleset has 847 active rules reviewed quarterly.",
    "IPv6 is enabled on all public-facing services since Q3 2025.",
    "Service mesh observability exports to Jaeger for distributed tracing.",
    "The network team uses Terraform for all infrastructure-as-code deployments.",
    "DDoS protection is provided by Cloudflare Spectrum on all ingress points.",
    "Internal DNS uses CoreDNS with etcd as the backend store.",
    "Network ACLs are version-controlled in the infra-policies Git repository.",
    "The jump host for SSH access is at 10.0.1.5 with session recording enabled.",
    "Traffic shaping limits video streaming to 100 Mbps aggregate during business hours.",
]

_INFRA_STORAGE_FACTS = [
    "Primary database storage uses NVMe SSDs with 3.2 TB capacity per node.",
    "Object storage is provided by MinIO running in erasure-coded mode with 12+4 parity.",
    "Backup retention policy keeps daily snapshots for 30 days and weekly for 1 year.",
    "The storage cluster has a total usable capacity of 480 TB across 3 racks.",
    "Tiered storage moves cold data to S3 Glacier after 90 days of inactivity.",
    "IOPS per volume is capped at 16,000 for gp3 EBS volumes.",
    "Block storage replication factor is 3 across availability zones.",
    "File shares use NFS v4.2 with Kerberos authentication.",
    "Storage encryption uses AES-256-XTS with keys managed by HashiCorp Vault.",
    "Deduplication ratio on backup storage averages 4.2:1 across all workloads.",
    "The Ceph cluster version is Reef (v18.2.1) with BlueStore backend.",
    "Snapshot-based cloning supports instant database refreshes for staging environments.",
    "Write-ahead log (WAL) storage uses dedicated 800 GB NVMe partitions.",
    "Storage latency SLA for transactional workloads is under 1 ms at p95.",
    "Quota enforcement limits each team namespace to 10 TB of persistent volume claims.",
    "Data lifecycle policies automatically expire temporary analysis datasets after 14 days.",
    "The SAN fabric uses 32 Gbps Fibre Channel with multipathing enabled.",
    "Storage performance metrics are exported to Prometheus every 15 seconds.",
    "Thin provisioning is enabled by default with 150% overcommit ratio.",
    "DR replication to the secondary site runs asynchronously with RPO of 15 minutes.",
]

_INFRA_COMPUTE_FACTS = [
    "Production Kubernetes cluster runs v1.29 on 48 worker nodes.",
    "Each worker node has 64 vCPUs and 256 GB RAM (m6i.16xlarge instances).",
    "Horizontal Pod Autoscaler targets 70% CPU utilization across all deployments.",
    "Node auto-scaling range is 48 to 120 nodes based on pending pod count.",
    "GPU workloads run on 8 dedicated p4d.24xlarge instances with A100 GPUs.",
    "Container runtime is containerd v1.7.8 with seccomp profiles enforced.",
    "Resource requests must equal limits for all production pods (guaranteed QoS).",
    "The control plane runs across 3 availability zones with etcd on dedicated nodes.",
    "Batch jobs use a separate Karpenter-managed node pool with spot instances.",
    "CPU steal time alert threshold is set at 5% sustained over 10 minutes.",
    "Memory overcommit ratio for non-production namespaces is 2:1.",
    "Pod disruption budgets require at least 2 replicas available during rolling updates.",
    "The admission controller enforces resource limits on all new pod deployments.",
    "Cluster DNS is CoreDNS with 3 replicas and anti-affinity scheduling.",
    "Init containers have a 120-second timeout before pods are marked as failed.",
    "CronJobs for data processing run on a dedicated node group with SSD storage.",
    "Taints and tolerations isolate PCI workloads to nodes in the compliance pool.",
    "kube-proxy runs in IPVS mode for better scalability than iptables.",
    "Ephemeral containers are enabled for debugging with limited RBAC access.",
    "Pod security standards enforce the restricted profile in all production namespaces.",
]

_INFRA_SECURITY_FACTS = [
    "Zero-trust architecture is enforced via SPIFFE/SPIRE for workload identity.",
    "Secret rotation period is 30 days for database credentials and 90 days for API keys.",
    "SIEM aggregation runs on Splunk Enterprise with 90-day hot storage retention.",
    "Vulnerability scanning uses Trivy in CI and Falco at runtime.",
    "SOC2 Type II audit was completed in November 2025 with zero findings.",
    "WAF rules block the OWASP Top 10 attack patterns on all public endpoints.",
    "Certificate management uses cert-manager with Let's Encrypt for public certs.",
    "SSH key authentication requires ed25519 keys with a minimum length of 256 bits.",
    "Security incident SLA requires acknowledgment within 15 minutes for P1 events.",
    "Penetration testing is conducted by NCC Group on a semi-annual basis.",
    "Container image signing uses Cosign with keys stored in KMS.",
    "OPA Gatekeeper policies enforce 47 security constraints in production clusters.",
    "Log integrity is guaranteed by immutable audit logs shipped to a WORM-compliant store.",
    "MFA is mandatory for all human access; service accounts use mTLS certificates.",
    "The bug bounty program pays up to $50,000 for critical vulnerabilities.",
    "Network IDS sensors monitor all inter-zone traffic for anomaly detection.",
    "Data classification labels (public, internal, confidential, restricted) are mandatory on all assets.",
    "Security training completion rate for engineering staff is 98% this quarter.",
    "Token lifetime for OAuth2 access tokens is 1 hour with refresh tokens valid for 24 hours.",
    "Incident response playbooks are stored in Confluence and tested quarterly via tabletop exercises.",
]

_INFRA_MONITORING_FACTS = [
    "Prometheus scrape interval is 15 seconds for application metrics.",
    "Grafana dashboards are organized into 12 team folders with 340 total panels.",
    "Alertmanager routes critical alerts to PagerDuty and informational to Slack #ops-alerts.",
    "The SLO for API availability is 99.95% measured over a rolling 30-day window.",
    "Distributed tracing sample rate is 1% for normal traffic and 100% for error traces.",
    "Log aggregation pipeline processes 2.3 TB per day through Fluent Bit to OpenSearch.",
    "Synthetic monitoring probes run every 60 seconds from 5 global regions.",
    "Error budget remaining for the billing service is 23 minutes this month.",
    "Custom metrics cardinality limit is 10,000 unique time series per namespace.",
    "On-call rotation uses a follow-the-sun model across US-West, EU, and APAC teams.",
    "Mean time to detect (MTTD) for P1 incidents averaged 3.2 minutes last quarter.",
    "Uptime Kuma monitors 89 internal and external HTTP endpoints.",
    "Cost allocation tags on cloud resources feed into the FinOps dashboard weekly.",
    "Anomaly detection uses a 3-sigma threshold on key business metrics.",
    "Black-box monitoring checks the full user registration flow every 5 minutes.",
    "Resource utilization reports are generated daily and sent to team leads at 08:00 UTC.",
    "The monitoring stack itself has an SLA of 99.99% with redundant Prometheus pairs.",
    "Alert fatigue reduction program decreased non-actionable alerts by 62% since Q1.",
    "Service dependency maps are auto-generated from tracing data in Kiali.",
    "Capacity planning forecasts are produced monthly using linear regression on utilization trends.",
]

SCENARIO_INFRA = HiveMindScenario(
    scenario_id="hive_infra",
    description="Infrastructure team: 5 agents covering networking, storage, compute, security, and monitoring domains.",
    num_agents=5,
    agent_domains={
        "networking": _INFRA_NETWORKING_FACTS,
        "storage": _INFRA_STORAGE_FACTS,
        "compute": _INFRA_COMPUTE_FACTS,
        "security": _INFRA_SECURITY_FACTS,
        "monitoring": _INFRA_MONITORING_FACTS,
    },
    questions=[
        # 5 single-domain questions
        HiveMindQuestion(
            question_id="hive_infra_q01",
            text="What firmware version is the primary load balancer running?",
            required_domains=["networking"],
            expected_keywords=["F5", "BIG-IP", "17.1.1"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_infra_q02",
            text="What is the backup retention policy for daily snapshots?",
            required_domains=["storage"],
            expected_keywords=["30", "days", "daily"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_infra_q03",
            text="How many worker nodes are in the production Kubernetes cluster?",
            required_domains=["compute"],
            expected_keywords=["48"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_infra_q04",
            text="When was the SOC2 Type II audit completed and what were the findings?",
            required_domains=["security"],
            expected_keywords=["November", "2025", "zero"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_infra_q05",
            text="What is the Prometheus scrape interval for application metrics?",
            required_domains=["monitoring"],
            expected_keywords=["15", "seconds"],
            difficulty="single_domain",
        ),
        # 5 cross-domain questions
        HiveMindQuestion(
            question_id="hive_infra_q06",
            text="How is the network segmented for PCI compliance, and what compute isolation supports it?",
            required_domains=["networking", "compute"],
            expected_keywords=["VLAN", "200", "taints", "tolerations", "compliance"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_infra_q07",
            text="Describe how storage encryption keys are managed and how secrets rotation works across the infrastructure.",
            required_domains=["storage", "security"],
            expected_keywords=["AES-256", "Vault", "30", "days", "rotation"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_infra_q08",
            text="What monitoring is in place for network latency SLAs between availability zones?",
            required_domains=["networking", "monitoring"],
            expected_keywords=["2", "ms", "SLO", "99.95"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_infra_q09",
            text="How does the compute auto-scaling interact with the monitoring alert thresholds?",
            required_domains=["compute", "monitoring"],
            expected_keywords=["70%", "CPU", "auto-scaling", "PagerDuty"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_infra_q10",
            text="What vulnerability scanning tools are used and how are container images secured before deployment to the cluster?",
            required_domains=["security", "compute"],
            expected_keywords=["Trivy", "Falco", "Cosign", "containerd"],
            difficulty="cross_domain",
        ),
        # 5 synthesis questions
        HiveMindQuestion(
            question_id="hive_infra_q11",
            text="A new microservice needs to be deployed that handles PCI card data. Walk through all the infrastructure requirements across networking, storage, compute, and security.",
            required_domains=["networking", "storage", "compute", "security"],
            expected_keywords=["VLAN", "200", "mTLS", "encryption", "AES-256", "compliance", "restricted"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_infra_q12",
            text="Describe the complete observability stack from network tracing through compute metrics to log aggregation and alerting.",
            required_domains=["networking", "compute", "monitoring"],
            expected_keywords=["Jaeger", "Prometheus", "Fluent Bit", "OpenSearch", "Grafana"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_infra_q13",
            text="What is the disaster recovery posture? Cover storage replication, compute redundancy, and security of the DR site.",
            required_domains=["storage", "compute", "security"],
            expected_keywords=["RPO", "15", "minutes", "availability zones", "3", "WORM"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_infra_q14",
            text="If the engineering team doubles in size, what infrastructure components need scaling? Consider network capacity, storage quotas, compute nodes, and monitoring cardinality.",
            required_domains=["networking", "storage", "compute", "monitoring"],
            expected_keywords=["500", "Mbps", "10", "TB", "120", "nodes", "10,000"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_infra_q15",
            text="Summarize how zero-trust security is implemented end-to-end: network layer, compute workload identity, storage access, and monitoring of security events.",
            required_domains=["networking", "storage", "compute", "security", "monitoring"],
            expected_keywords=["mTLS", "SPIFFE", "Kerberos", "zero-trust", "SIEM", "Splunk"],
            difficulty="synthesis",
        ),
    ],
)


# ============================================================
# Scenario 2: Software Architecture
# ============================================================

_ARCH_FRONTEND_FACTS = [
    "The web application uses React 18 with TypeScript and Vite as the build tool.",
    "State management is handled by Zustand with persistence middleware for offline support.",
    "The design system has 84 components published as @company/ui on the internal npm registry.",
    "Server-side rendering is provided by Next.js 14 with incremental static regeneration.",
    "E2E tests use Playwright with 340 test scenarios covering all critical user flows.",
    "The bundle size budget is 250 KB gzipped for the main chunk.",
    "Accessibility compliance targets WCAG 2.1 AA with automated aXe scans in CI.",
    "Internationalization supports 12 languages using react-intl with ICU message format.",
    "The micro-frontend architecture uses Module Federation for lazy-loaded features.",
    "CSS is managed with Tailwind CSS v3.4 using a custom design token preset.",
    "Real-time features use WebSocket connections via Socket.io with automatic reconnection.",
    "Feature flags are managed through LaunchDarkly with server-side evaluation.",
    "Image optimization uses next/image with AVIF format preference and CDN caching.",
    "Error tracking integrates with Sentry with a 10% sample rate in production.",
    "The frontend monorepo uses Turborepo for parallel builds across 7 workspace packages.",
    "Authentication flow uses PKCE-enhanced OAuth2 with silent token refresh.",
    "Performance monitoring tracks Core Web Vitals with a LCP target under 2.5 seconds.",
    "API calls use TanStack Query (React Query) with stale-while-revalidate caching.",
    "The component storybook has 420 stories and is deployed to an internal URL.",
    "Mobile responsiveness breakpoints follow the design system at 640, 768, 1024, and 1280 pixels.",
]

_ARCH_BACKEND_FACTS = [
    "The API is built with FastAPI v0.108 running on Python 3.12.",
    "Request validation uses Pydantic v2 models with strict mode enabled.",
    "The service mesh routes traffic through Envoy sidecar proxies.",
    "Rate limiting is implemented at the API gateway level: 1000 req/min per client.",
    "Background task processing uses Celery with Redis as the broker.",
    "API versioning follows the URL-based strategy: /api/v1/, /api/v2/.",
    "Health checks expose /healthz (liveness) and /readyz (readiness) endpoints.",
    "OpenAPI documentation is auto-generated and hosted at /docs with Swagger UI.",
    "Structured logging uses structlog with JSON output to stdout.",
    "Circuit breaker pattern is implemented using tenacity with 5-failure threshold.",
    "gRPC is used for inter-service communication between backend microservices.",
    "The event bus uses Apache Kafka with 12 partitions per topic.",
    "Idempotency keys are required for all mutation endpoints to prevent duplicate operations.",
    "Request tracing propagates trace-id headers through all downstream calls.",
    "The backend serves 45,000 requests per second at peak with p99 latency under 200ms.",
    "Data serialization uses MessagePack for internal service communication.",
    "Caching strategy uses Redis with a 5-minute TTL for frequently accessed entities.",
    "API deprecation policy requires 6-month notice with sunset headers on responses.",
    "Webhooks are delivered with exponential backoff: 1s, 5s, 30s, 5m, 30m retries.",
    "Service discovery uses Consul with health-check-based deregistration.",
]

_ARCH_DATABASE_FACTS = [
    "Primary datastore is PostgreSQL 16 with logical replication to 2 read replicas.",
    "Connection pooling uses PgBouncer with 500 max connections in transaction mode.",
    "Database migrations run through Alembic with a CI gate that rejects backward-incompatible changes.",
    "Full-text search uses PostgreSQL tsvector with GIN indexes on content columns.",
    "The analytics warehouse is Snowflake with daily CDC replication from production.",
    "Redis 7.2 serves as the caching layer with 64 GB allocated across 3 sentinel nodes.",
    "Time-series data for metrics storage uses TimescaleDB hypertables with 7-day chunk intervals.",
    "Database query performance is monitored via pg_stat_statements with a 10ms slow query threshold.",
    "Schema design follows a multi-tenant model with row-level security policies per tenant.",
    "The event store uses append-only tables with JSONB payloads and event_type indexing.",
    "Backup strategy uses pg_basebackup for physical backups and WAL archiving for PITR.",
    "Table partitioning by date is applied to audit_log and event_log tables.",
    "Connection encryption uses TLS 1.3 with certificates rotated by cert-manager.",
    "Database vacuum settings use autovacuum with a scale factor of 0.1 for large tables.",
    "The ORM is SQLAlchemy 2.0 with async session support via asyncpg driver.",
    "Index bloat monitoring triggers reindex when bloat exceeds 30%.",
    "Read replicas serve analytics queries with a max replication lag alert at 5 seconds.",
    "Test data seeding uses factory_boy with 50 deterministic fixtures per entity type.",
    "Database credentials rotate every 30 days via Vault dynamic secrets.",
    "The ERD has 127 tables across 8 bounded context schemas.",
]

_ARCH_DEVOPS_FACTS = [
    "CI/CD pipeline runs on GitHub Actions with 12 parallel runners.",
    "Container images are built with multi-stage Dockerfiles and pushed to ECR.",
    "Deployment strategy uses blue-green with automatic rollback on health check failure.",
    "Infrastructure is managed with Terraform v1.7 using remote state in S3.",
    "Helm charts manage Kubernetes manifests with values overrides per environment.",
    "The staging environment mirrors production at 25% scale with synthetic traffic.",
    "Deployment frequency is 8-12 times per day to production.",
    "Feature branch environments spin up automatically on PR creation and tear down on merge.",
    "Secret management uses External Secrets Operator syncing from AWS Secrets Manager.",
    "Build caching reduces average CI time from 12 minutes to 4 minutes.",
    "Canary deployments route 5% of traffic to new versions for 10 minutes before full rollout.",
    "GitOps workflow uses ArgoCD for declarative cluster state management.",
    "Artifact scanning with Snyk blocks deployments with critical CVEs.",
    "The release process follows semantic versioning with automated changelog generation.",
    "Environment parity is enforced via OPA policies comparing staging and production configs.",
    "Rollback time from detection to complete revert averages 90 seconds.",
    "Database migrations run as pre-deploy jobs with automatic rollback scripts.",
    "Load testing uses k6 with 500 virtual users simulating peak traffic patterns.",
    "The deployment pipeline has 7 stages: lint, test, build, scan, deploy-staging, verify, deploy-prod.",
    "Post-deployment verification runs smoke tests against 15 critical API endpoints.",
]

_ARCH_TESTING_FACTS = [
    "Unit test coverage target is 85% for all backend services.",
    "Integration tests run against a dockerized PostgreSQL and Redis in CI.",
    "Mutation testing uses mutmut with a kill rate target of 70%.",
    "Contract testing between services uses Pact with a central broker.",
    "Performance regression tests compare p95 latency against baseline within 10% tolerance.",
    "Test data generation uses Hypothesis for property-based testing of API endpoints.",
    "Visual regression testing uses Chromatic snapshots on every PR.",
    "API conformance tests validate all endpoints against the OpenAPI specification.",
    "The test suite runs 4,200 tests in 6 minutes using pytest-xdist with 8 workers.",
    "Flaky test detection quarantines tests with >5% failure rate over 50 runs.",
    "Chaos engineering experiments run weekly using Litmus with pod-kill and network-delay scenarios.",
    "Security testing includes DAST scans with ZAP on staging before each production deploy.",
    "Accessibility tests run aXe-core checks on all pages during E2E test execution.",
    "Test environments use testcontainers for isolated, reproducible integration tests.",
    "Code review requires at least one passing test run and coverage report before merge.",
    "Snapshot testing captures API response schemas to detect unintended breaking changes.",
    "Load test results are archived and compared across releases for trend analysis.",
    "The QA team maintains 180 manual test cases for features requiring human judgment.",
    "Test pyramid ratio is 60% unit, 25% integration, 10% E2E, 5% manual.",
    "Feature flag tests verify behavior for both enabled and disabled states of every flag.",
]

SCENARIO_ARCH = HiveMindScenario(
    scenario_id="hive_arch",
    description="Software architecture team: 5 agents covering frontend, backend, database, devops, and testing domains.",
    num_agents=5,
    agent_domains={
        "frontend": _ARCH_FRONTEND_FACTS,
        "backend": _ARCH_BACKEND_FACTS,
        "database": _ARCH_DATABASE_FACTS,
        "devops": _ARCH_DEVOPS_FACTS,
        "testing": _ARCH_TESTING_FACTS,
    },
    questions=[
        # 5 single-domain
        HiveMindQuestion(
            question_id="hive_arch_q01",
            text="What build tool does the frontend use and what is the bundle size budget?",
            required_domains=["frontend"],
            expected_keywords=["Vite", "250", "KB"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_arch_q02",
            text="What framework and Python version powers the backend API?",
            required_domains=["backend"],
            expected_keywords=["FastAPI", "Python", "3.12"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_arch_q03",
            text="How many tables are in the database ERD and what version of PostgreSQL is used?",
            required_domains=["database"],
            expected_keywords=["127", "PostgreSQL", "16"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_arch_q04",
            text="What is the deployment frequency and what strategy is used?",
            required_domains=["devops"],
            expected_keywords=["8-12", "blue-green"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_arch_q05",
            text="What is the test pyramid ratio and total number of automated tests?",
            required_domains=["testing"],
            expected_keywords=["60%", "unit", "4,200"],
            difficulty="single_domain",
        ),
        # 5 cross-domain
        HiveMindQuestion(
            question_id="hive_arch_q06",
            text="How does the frontend communicate with the backend API, and how is caching handled on both sides?",
            required_domains=["frontend", "backend"],
            expected_keywords=["TanStack", "React Query", "Redis", "5-minute", "stale-while-revalidate"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_arch_q07",
            text="Describe how database migrations are handled in the CI/CD pipeline.",
            required_domains=["database", "devops"],
            expected_keywords=["Alembic", "pre-deploy", "rollback", "backward-incompatible"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_arch_q08",
            text="What contract testing exists between frontend and backend services?",
            required_domains=["backend", "testing"],
            expected_keywords=["Pact", "OpenAPI", "conformance"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_arch_q09",
            text="How are feature flags managed and how are they tested?",
            required_domains=["frontend", "testing"],
            expected_keywords=["LaunchDarkly", "enabled", "disabled", "flag"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_arch_q10",
            text="How does the deployment pipeline ensure database and backend compatibility?",
            required_domains=["database", "devops"],
            expected_keywords=["Alembic", "canary", "health check", "rollback"],
            difficulty="cross_domain",
        ),
        # 5 synthesis
        HiveMindQuestion(
            question_id="hive_arch_q11",
            text="Trace the complete path of a user request from frontend button click through backend processing, database query, and back. Include caching, tracing, and error handling at each layer.",
            required_domains=["frontend", "backend", "database"],
            expected_keywords=["React", "FastAPI", "PostgreSQL", "Redis", "trace-id", "Sentry"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_arch_q12",
            text="Describe the complete quality assurance pipeline from code commit through deployment. Cover linting, testing, scanning, staging verification, and production rollout.",
            required_domains=["devops", "testing", "backend"],
            expected_keywords=["GitHub Actions", "pytest", "Snyk", "staging", "canary", "smoke"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_arch_q13",
            text="How would you add a new feature that requires a frontend component, backend endpoint, database table, deployment config, and test coverage?",
            required_domains=["frontend", "backend", "database", "devops", "testing"],
            expected_keywords=["React", "FastAPI", "Alembic", "Helm", "Pact"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_arch_q14",
            text="Summarize the observability and monitoring strategy across all layers: frontend performance, backend tracing, database monitoring, and deployment health checks.",
            required_domains=["frontend", "backend", "database", "devops"],
            expected_keywords=["Core Web Vitals", "structlog", "pg_stat_statements", "ArgoCD"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_arch_q15",
            text="How is security implemented across the stack? Cover frontend auth, backend rate limiting, database encryption, and deployment scanning.",
            required_domains=["frontend", "backend", "database", "devops"],
            expected_keywords=["PKCE", "OAuth2", "rate", "TLS", "1.3", "Snyk"],
            difficulty="synthesis",
        ),
    ],
)


# ============================================================
# Scenario 3: Incident Response
# ============================================================

_INCIDENT_TIMELINE_FACTS = [
    "2026-02-15 14:23 UTC: First automated alert fired - payment processing latency spike.",
    "2026-02-15 14:25 UTC: On-call engineer Alice acknowledged the PagerDuty alert.",
    "2026-02-15 14:28 UTC: Alice declared Incident INC-4521 with severity P1.",
    "2026-02-15 14:30 UTC: Incident channel #inc-4521 created in Slack.",
    "2026-02-15 14:35 UTC: Initial assessment: payment API returning 503 errors at 40% rate.",
    "2026-02-15 14:42 UTC: Engineering VP Bob joined the incident as executive sponsor.",
    "2026-02-15 14:50 UTC: Decision to roll back the 14:00 deployment of payment-service v2.8.1.",
    "2026-02-15 14:55 UTC: Rollback to payment-service v2.7.9 initiated.",
    "2026-02-15 15:02 UTC: Rollback complete. Error rate dropped from 40% to 8%.",
    "2026-02-15 15:10 UTC: Residual errors traced to stale database connection pool.",
    "2026-02-15 15:15 UTC: Database connection pool recycled. Error rate dropped to 0.1%.",
    "2026-02-15 15:30 UTC: Monitoring confirmed stable for 15 minutes. P1 downgraded to P3.",
    "2026-02-15 15:45 UTC: Customer support reported 127 customer complaints received during the incident.",
    "2026-02-15 16:00 UTC: Incident officially resolved. Total duration: 97 minutes.",
    "2026-02-15 16:30 UTC: Post-incident review meeting scheduled for 2026-02-17.",
    "2026-02-17 10:00 UTC: Post-mortem identified root cause as database migration in v2.8.1.",
    "2026-02-17 10:30 UTC: Action items: add migration dry-run step, improve canary detection.",
    "2026-02-17 11:00 UTC: 5 action items assigned with owners and due dates.",
    "The incident impacted approximately 12,000 transactions over the 97-minute window.",
    "Financial impact estimated at $340,000 in delayed or failed payment processing.",
]

_INCIDENT_LOGS_FACTS = [
    "payment-service logs show: ERROR ConnectionPool exhausted - all 50 connections in use.",
    "Database logs show: WARNING too many connections for role 'payment_svc' - limit 50 reached.",
    "The v2.8.1 deployment added a new database migration: ALTER TABLE payments ADD COLUMN metadata JSONB.",
    "Migration locked the payments table for 4 minutes and 12 seconds during the ALTER.",
    "During the table lock, connection pool filled as queries queued behind the lock.",
    "After lock released, pool connections were in a bad state due to timeout-killed queries.",
    "PgBouncer logs show: ERROR server_login failed for payment_svc - too many connections.",
    "Application retry logic created a thundering herd when the pool drained.",
    "Log volume spiked to 45,000 lines per minute during the incident (normal: 3,000).",
    "Gateway logs show: 503 Service Unavailable for /api/v2/payments/* endpoints only.",
    "The /api/v2/payments/health endpoint returned 200 throughout (it uses a separate connection).",
    "Stack traces show SQLAlchemy OperationalError: cannot acquire connection within 30 seconds.",
    "Celery worker logs show: 847 payment processing tasks stuck in pending state.",
    "Log timestamps reveal a 7-second gap in payment-service logs between 14:26 and 14:33 UTC.",
    "Debug logs reveal the migration ran without --lock-timeout flag.",
    "The previous deployment (v2.7.9) had no database migration component.",
    "Container restart logs show payment-service pods restarted 3 times due to OOM during the incident.",
    "Memory profiling shows leaked connection objects accumulated 2.1 GB before OOM kill.",
    "Kubernetes events show: Pod evicted due to memory pressure on node worker-17.",
    "Audit log confirms the deployment was triggered by CI pipeline run #8847.",
]

_INCIDENT_METRICS_FACTS = [
    "Payment API p99 latency jumped from 180ms to 45,000ms (45 seconds) at 14:23 UTC.",
    "Error rate on /api/v2/payments/* went from 0.01% to 40.3% within 2 minutes.",
    "Database active connections hit the maximum of 50 at 14:24 UTC and stayed saturated for 38 minutes.",
    "CPU utilization on payment-service pods spiked to 94% due to retry storms.",
    "Memory usage on payment-service pods grew from 1.2 GB to 3.8 GB before OOM.",
    "Kafka consumer lag for payment-events topic grew to 12,400 messages.",
    "Successful transaction throughput dropped from 2,100 TPS to 340 TPS.",
    "After rollback at 15:02, error rate dropped to 8% within 30 seconds.",
    "After pool recycle at 15:15, latency returned to normal p99 of 180ms.",
    "The canary phase (5% traffic) showed no elevated errors because the migration only runs once.",
    "Grafana dashboard 'Payment Health' was the first to show the anomaly.",
    "SLO burn rate exceeded 100x the budget for the 1-hour window.",
    "Network I/O on database primary increased 3x due to retry traffic.",
    "Prometheus recorded 4,847 firing alerts during the 97-minute window.",
    "The payment success rate SLI dropped to 59.7% (target: 99.95%).",
    "Disk I/O on the database primary spiked to 95% utilization during the ALTER.",
    "Worker node CPU steal time stayed at 0%, ruling out noisy neighbor issues.",
    "DNS resolution latency remained normal at 2ms, ruling out DNS as a factor.",
    "Redis cache hit rate for payment sessions dropped from 92% to 34%.",
    "Load balancer connection queue depth reached 8,400 (normal: under 100).",
]

_INCIDENT_CODE_FACTS = [
    "The v2.8.1 release contained 14 commits: 11 feature commits and 3 migration commits.",
    "The migration script used ALTER TABLE ADD COLUMN without CONCURRENTLY option.",
    "PostgreSQL ALTER TABLE ADD COLUMN with a DEFAULT requires an exclusive lock on the table.",
    "The migration could have used ADD COLUMN ... DEFAULT ... followed by backfill to avoid locking.",
    "Connection pool configuration was set to pool_size=50, max_overflow=0 (no overflow allowed).",
    "The pool_pre_ping setting was disabled, meaning stale connections were not detected.",
    "Retry logic used constant 1-second intervals instead of exponential backoff.",
    "The circuit breaker threshold was set to 50 failures (should have been 5 for this service).",
    "The payments table has 47 million rows, making ALTER TABLE operations slow.",
    "A previous PR (#3847) had suggested using pg_repack for schema changes but was not merged.",
    "The deployment pipeline did not include a migration dry-run step.",
    "Code review for the migration PR (#3891) was approved in 12 minutes by a single reviewer.",
    "The reviewer's comment: 'LGTM, standard column addition' missed the locking implication.",
    "Feature flag 'use_payment_metadata' was supposed to guard the new column but was enabled by default.",
    "The health check endpoint used a lightweight SQL query (SELECT 1) that bypassed the payments table.",
    "Rollback script existed but required manual SSH access; it was later automated.",
    "The migration was tested against a staging database with only 500,000 rows (1% of production).",
    "Staging did not reproduce the issue because the lock completed in under 1 second.",
    "The connection pool's recycle setting was 3600 seconds (too long for recovering from pool exhaustion).",
    "Post-incident fix: pool_pre_ping=True, max_overflow=10, recycle=300, backoff=exponential.",
]

_INCIDENT_COMMS_FACTS = [
    "Status page was updated at 14:32 UTC: 'Investigating payment processing delays.'",
    "Customer support sent the first batch email at 14:40 UTC to affected merchants.",
    "Engineering posted in #inc-4521: 'Root cause identified - database migration issue.'",
    "VP Bob sent executive summary to C-suite at 14:45 UTC.",
    "Status page updated at 15:05 UTC: 'Fix deployed. Monitoring for stability.'",
    "Customer support handled 127 tickets during the incident; average response time 8 minutes.",
    "Social media team posted on Twitter/X at 14:50 UTC acknowledging the issue.",
    "Partner API users were notified via automated webhook status updates.",
    "Internal communication followed the DACI framework: Driver=Alice, Approver=Bob.",
    "Post-mortem document was shared company-wide within 48 hours.",
    "The blameless post-mortem identified 5 contributing factors and 5 action items.",
    "Action item owners: Alice (migration dry-run), Charlie (canary improvement), Dave (pool config).",
    "Customer credits totaling $15,200 were issued to merchants with failed transactions.",
    "Legal team was notified due to potential SLA breach with 3 enterprise customers.",
    "SLA breach notifications were sent to 3 enterprise customers within 24 hours.",
    "The incident report was reviewed by the VP of Engineering, CTO, and Head of Customer Success.",
    "A follow-up all-hands included a 'lessons learned' segment on safe migration practices.",
    "The incident response playbook was updated to include database migration-specific steps.",
    "Communication templates for payment outages were added to the incident response toolkit.",
    "Quarterly review flagged this as the highest-impact incident of Q1 2026.",
]

SCENARIO_INCIDENT = HiveMindScenario(
    scenario_id="hive_incident",
    description="Incident response team: 5 agents covering timeline, logs, metrics, code changes, and communications.",
    num_agents=5,
    agent_domains={
        "timeline": _INCIDENT_TIMELINE_FACTS,
        "logs": _INCIDENT_LOGS_FACTS,
        "metrics": _INCIDENT_METRICS_FACTS,
        "code": _INCIDENT_CODE_FACTS,
        "comms": _INCIDENT_COMMS_FACTS,
    },
    questions=[
        # 5 single-domain
        HiveMindQuestion(
            question_id="hive_inc_q01",
            text="At what time was the incident declared and what severity was assigned?",
            required_domains=["timeline"],
            expected_keywords=["14:28", "P1", "INC-4521"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_inc_q02",
            text="What error messages appeared in the payment-service logs during the incident?",
            required_domains=["logs"],
            expected_keywords=["ConnectionPool", "exhausted", "50"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_inc_q03",
            text="What was the peak error rate and how long did database connections stay saturated?",
            required_domains=["metrics"],
            expected_keywords=["40.3%", "38", "minutes"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_inc_q04",
            text="What code change in v2.8.1 caused the incident?",
            required_domains=["code"],
            expected_keywords=["ALTER TABLE", "ADD COLUMN", "metadata", "JSONB"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_inc_q05",
            text="How many customer complaints were received and what credits were issued?",
            required_domains=["comms"],
            expected_keywords=["127", "$15,200"],
            difficulty="single_domain",
        ),
        # 5 cross-domain
        HiveMindQuestion(
            question_id="hive_inc_q06",
            text="Correlate the timeline events with the metric changes. When did the rollback happen and how did metrics respond?",
            required_domains=["timeline", "metrics"],
            expected_keywords=["14:55", "15:02", "8%", "rollback"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_inc_q07",
            text="What do the logs and code reveal about why the connection pool was exhausted?",
            required_domains=["logs", "code"],
            expected_keywords=["ALTER TABLE", "lock", "pool_size=50", "max_overflow=0"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_inc_q08",
            text="How did the communications team respond relative to the incident timeline?",
            required_domains=["timeline", "comms"],
            expected_keywords=["14:32", "status page", "14:40", "merchants"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_inc_q09",
            text="What metrics confirmed the canary deployment missed the issue, and what code explains why?",
            required_domains=["metrics", "code"],
            expected_keywords=["5%", "canary", "migration", "once", "staging", "500,000"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_inc_q10",
            text="What action items came from the post-mortem and how were they communicated?",
            required_domains=["timeline", "comms"],
            expected_keywords=["5", "action items", "migration dry-run", "canary", "blameless"],
            difficulty="cross_domain",
        ),
        # 5 synthesis
        HiveMindQuestion(
            question_id="hive_inc_q11",
            text="Reconstruct the complete incident from first alert to resolution using evidence from all five domains.",
            required_domains=["timeline", "logs", "metrics", "code", "comms"],
            expected_keywords=["14:23", "ALTER TABLE", "40.3%", "rollback", "v2.7.9", "127"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_inc_q12",
            text="What was the root cause chain? Start with the code change, explain how it caused the log errors, what the metrics showed, and how it was ultimately resolved.",
            required_domains=["code", "logs", "metrics", "timeline"],
            expected_keywords=["migration", "lock", "ConnectionPool", "45,000ms", "rollback", "pool recycle"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_inc_q13",
            text="What systemic failures allowed this incident to happen? Cover code review, testing, deployment pipeline, monitoring, and communication gaps.",
            required_domains=["code", "metrics", "comms", "timeline", "logs"],
            expected_keywords=["12 minutes", "single reviewer", "staging", "500,000", "canary"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_inc_q14",
            text="Calculate the total business impact: financial loss, customer impact, SLA breaches, and operational costs.",
            required_domains=["timeline", "metrics", "comms"],
            expected_keywords=["$340,000", "12,000", "transactions", "$15,200", "3", "enterprise"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_inc_q15",
            text="What specific changes should be made to prevent a recurrence? Reference concrete configuration values, code changes, and process improvements from the evidence.",
            required_domains=["code", "logs", "metrics", "comms", "timeline"],
            expected_keywords=["pool_pre_ping", "max_overflow=10", "exponential", "dry-run", "playbook"],
            difficulty="synthesis",
        ),
    ],
)


# ============================================================
# Scenario 4: Research Synthesis
# ============================================================

_RESEARCH_PAPER_A_FACTS = [
    "Paper A: 'Scaling Laws for Neural Retrieval' by Zhang et al., published in ICML 2026.",
    "Paper A found that retrieval accuracy scales as a power law with index size up to 10B documents.",
    "Paper A's key result: doubling index size improves recall@10 by 8.3% on average.",
    "Paper A used the MS MARCO v3 benchmark with 15M passage collection.",
    "Paper A's model architecture: dual-encoder with 768-dimensional embeddings.",
    "Paper A reports that dense retrieval outperforms BM25 by 23% at the 10B document scale.",
    "Paper A's training used contrastive learning with hard negative mining from BM25 results.",
    "Paper A observed diminishing returns in recall improvement beyond 50B index size.",
    "Paper A's compute cost: 512 A100-hours for training, 2048 A100-hours for index building.",
    "Paper A recommends hybrid dense+sparse retrieval for optimal cost-quality tradeoff.",
    "Paper A's evaluation: tested on 5 languages (EN, ZH, DE, FR, JA) with consistent scaling.",
    "Paper A found that query latency scales logarithmically with index size using HNSW.",
    "Paper A's ablation shows embedding dimension is the strongest predictor of retrieval quality.",
    "Paper A identifies a phase transition in retrieval quality at approximately 1B index entries.",
    "Paper A's efficiency metric: 0.83 recall@10 at 95th percentile latency of 12ms.",
    "Paper A used product quantization to reduce memory from 3TB to 380GB for 10B vectors.",
    "Paper A found that fine-tuning on domain-specific data improves recall by 15% over zero-shot.",
    "Paper A's negative result: re-ranking with cross-encoders adds latency without improving top-1.",
    "Paper A's training data: 800M query-passage pairs mined from search logs.",
    "Paper A concludes that retrieval scaling is more cost-effective than LLM scaling for RAG systems.",
]

_RESEARCH_PAPER_B_FACTS = [
    "Paper B: 'Memory-Augmented Generation with Episodic Buffers' by Kim & Patel, NeurIPS 2025.",
    "Paper B introduces the Episodic Buffer Architecture (EBA) for long-context generation.",
    "Paper B's key insight: episodic memory slots outperform sliding window attention for 100K+ tokens.",
    "Paper B's benchmark: LongBench-v2 with passages up to 500K tokens.",
    "Paper B achieves 91.2% accuracy on 128K-token tasks vs 73.4% for standard transformers.",
    "Paper B's architecture uses 256 episodic memory slots with learned gating mechanisms.",
    "Paper B shows that episodic buffers reduce memory usage by 4x compared to full attention.",
    "Paper B's training: 2-stage process - pre-train on BookCorpus, fine-tune on long-context tasks.",
    "Paper B identifies catastrophic forgetting in standard models beyond 64K context length.",
    "Paper B's solution: write-once-read-many (WORM) episodic slots prevent overwriting.",
    "Paper B's compute cost: 128 H100-hours for the full training pipeline.",
    "Paper B found that random slot allocation performs nearly as well as learned allocation.",
    "Paper B compares against Mamba, RWKV, and Hyena architectures on long-context benchmarks.",
    "Paper B shows EBA maintains consistent performance from 4K to 512K tokens without degradation.",
    "Paper B's efficiency: 2.1x faster inference than standard attention at 256K context length.",
    "Paper B's ablation: removing the gating mechanism reduces accuracy by 12% on multi-hop tasks.",
    "Paper B found that 256 slots is optimal; 128 underperforms and 512 shows no improvement.",
    "Paper B's negative finding: EBA underperforms on tasks requiring precise positional information.",
    "Paper B released an open-source implementation achieving 85% of the paper's reported accuracy.",
    "Paper B concludes episodic buffers are complementary to, not a replacement for, retrieval systems.",
]

_RESEARCH_PAPER_C_FACTS = [
    "Paper C: 'Federated Knowledge Graphs for Distributed AI' by Okonkwo et al., AAAI 2026.",
    "Paper C proposes FedKG: a protocol for merging knowledge graphs across organizational boundaries.",
    "Paper C's key result: FedKG achieves 94% entity alignment accuracy without sharing raw data.",
    "Paper C benchmark: evaluated on Wikidata-5M, Freebase-15K, and a new enterprise KG dataset.",
    "Paper C's architecture: federated learning + graph neural networks for entity matching.",
    "Paper C shows that knowledge graph completion improves 31% when merging 5 organizational KGs.",
    "Paper C addresses privacy via differential privacy with epsilon=1.0 noise injection.",
    "Paper C's communication cost: 12MB per round for organizations with 1M-entity KGs.",
    "Paper C found that 10 federation rounds are sufficient for convergence in most cases.",
    "Paper C's entity resolution handles multilingual entities with 89% accuracy across 8 languages.",
    "Paper C reports 3.2x query coverage improvement over isolated organizational KGs.",
    "Paper C's negative finding: temporal relations (before/after) degrade to 71% after federation.",
    "Paper C uses homomorphic encryption for sensitive attribute matching.",
    "Paper C's evaluation includes a healthcare KG federation with 3 hospital systems.",
    "Paper C found that schema alignment is the primary bottleneck: 47% of federation time.",
    "Paper C recommends starting with entity types that have high overlap across organizations.",
    "Paper C's scalability: tested up to 50 participating organizations with linear communication cost.",
    "Paper C identifies trust scoring: organizations with higher data quality improve global KG more.",
    "Paper C's open-source release includes reference implementations for PyTorch Geometric.",
    "Paper C concludes that federated KGs enable cross-organizational AI without data centralization.",
]

_RESEARCH_PAPER_D_FACTS = [
    "Paper D: 'Continual Learning in Production Agents' by Morales & Tanaka, ICLR 2026.",
    "Paper D studies how deployed agents can learn from user interactions without catastrophic forgetting.",
    "Paper D's key result: Elastic Weight Consolidation (EWC+) reduces forgetting by 67% vs naive fine-tuning.",
    "Paper D benchmark: CL-Agent-Bench with 50 sequential task distributions over 6 months.",
    "Paper D's architecture: adapter-based learning with frozen base model and task-specific LoRA heads.",
    "Paper D shows that replay buffers of 10K examples maintain 95% of original task performance.",
    "Paper D's production study: deployed at a customer support center with 500K daily interactions.",
    "Paper D found that user feedback signals (thumbs up/down) are noisy but usable with 0.72 label accuracy.",
    "Paper D's training cost: incremental updates cost 0.3% of full fine-tuning compute.",
    "Paper D identifies a critical window: agents must adapt within 48 hours of distribution shift.",
    "Paper D compares EWC+, PackNet, Progressive Neural Networks, and simple fine-tuning.",
    "Paper D shows that task detection (knowing when the task has changed) is unsolved: 61% accuracy.",
    "Paper D's negative finding: continual learning degrades on tasks requiring precise numerical recall.",
    "Paper D found that knowledge distillation from the production model to a student improves stability.",
    "Paper D's data efficiency: 500 examples per new task achieves 80% of full-data performance.",
    "Paper D recommends separate memory banks for factual knowledge vs procedural skills.",
    "Paper D's evaluation includes both automated metrics and human preference judgments (N=1200).",
    "Paper D found that prompt-based continual learning (no weight updates) works for simple domain shifts.",
    "Paper D's open-source toolkit 'CL-Deploy' includes reference implementations and benchmarks.",
    "Paper D concludes production continual learning is viable but requires careful monitoring of forgetting.",
]

_RESEARCH_PAPER_E_FACTS = [
    "Paper E: 'Benchmarking Multi-Agent Coordination' by Nakamura et al., ACL 2026.",
    "Paper E introduces CoordBench: a benchmark for evaluating multi-agent communication protocols.",
    "Paper E's key finding: agents with shared memory outperform message-passing by 41% on complex tasks.",
    "Paper E benchmark: 200 collaborative tasks across planning, coding, analysis, and writing domains.",
    "Paper E's taxonomy: broadcast, gossip, hierarchical, and shared-memory communication patterns.",
    "Paper E shows that 5 agents is optimal; 3 underperforms and 7+ shows coordination overhead.",
    "Paper E's cost analysis: shared-memory coordination costs 2.3x a single agent but yields 3.1x output quality.",
    "Paper E found that role specialization improves task completion by 28% over homogeneous agents.",
    "Paper E identifies the 'echo chamber' problem: agents converge on wrong answers 15% of the time.",
    "Paper E's solution to echo chamber: adversarial agent that challenges consensus reduces errors by 40%.",
    "Paper E compares GPT-4, Claude 3, and Gemini Pro as base models for multi-agent systems.",
    "Paper E shows that inter-agent communication tokens represent 34% of total token usage.",
    "Paper E's negative finding: hierarchical coordination (manager+workers) underperforms flat gossip.",
    "Paper E found that asynchronous coordination outperforms synchronous by 18% due to reduced blocking.",
    "Paper E's latency analysis: shared-memory adds 120ms overhead per agent interaction.",
    "Paper E recommends gossip protocols with 3 rounds for optimal information dissemination.",
    "Paper E's evaluation includes both automated scoring and expert human annotation (N=500).",
    "Paper E found that agents with access to a shared scratchpad produce 25% more coherent outputs.",
    "Paper E's open dataset: 200 tasks with ground truth, agent traces, and human preference labels.",
    "Paper E concludes shared memory is the most promising coordination mechanism for complex tasks.",
]

SCENARIO_RESEARCH = HiveMindScenario(
    scenario_id="hive_research",
    description="Research synthesis: 5 agents each read a different paper on AI systems, must synthesize cross-paper findings.",
    num_agents=5,
    agent_domains={
        "paper_a": _RESEARCH_PAPER_A_FACTS,
        "paper_b": _RESEARCH_PAPER_B_FACTS,
        "paper_c": _RESEARCH_PAPER_C_FACTS,
        "paper_d": _RESEARCH_PAPER_D_FACTS,
        "paper_e": _RESEARCH_PAPER_E_FACTS,
    },
    questions=[
        # 5 single-domain
        HiveMindQuestion(
            question_id="hive_res_q01",
            text="What is the key scaling result from Paper A regarding index size and retrieval accuracy?",
            required_domains=["paper_a"],
            expected_keywords=["power law", "8.3%", "doubling"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_res_q02",
            text="What is the Episodic Buffer Architecture and how many memory slots does it use?",
            required_domains=["paper_b"],
            expected_keywords=["EBA", "256", "slots", "episodic"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_res_q03",
            text="How does FedKG handle privacy when merging knowledge graphs?",
            required_domains=["paper_c"],
            expected_keywords=["differential privacy", "epsilon=1.0", "homomorphic"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_res_q04",
            text="What continual learning method does Paper D recommend and how much does it reduce forgetting?",
            required_domains=["paper_d"],
            expected_keywords=["EWC+", "67%", "forgetting"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_res_q05",
            text="According to Paper E, what is the optimal number of agents and why?",
            required_domains=["paper_e"],
            expected_keywords=["5", "optimal", "coordination overhead"],
            difficulty="single_domain",
        ),
        # 5 cross-domain
        HiveMindQuestion(
            question_id="hive_res_q06",
            text="How do Paper A's retrieval system and Paper B's episodic buffers complement each other for RAG?",
            required_domains=["paper_a", "paper_b"],
            expected_keywords=["retrieval", "episodic", "complementary", "long-context"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_res_q07",
            text="Compare the compute costs reported across Paper A, Paper B, and Paper D.",
            required_domains=["paper_a", "paper_b", "paper_d"],
            expected_keywords=["512", "A100", "128", "H100", "0.3%"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_res_q08",
            text="What does Paper C's federation approach share with Paper E's multi-agent coordination findings?",
            required_domains=["paper_c", "paper_e"],
            expected_keywords=["shared", "memory", "communication", "federation"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_res_q09",
            text="How might Paper D's continual learning approach address Paper B's catastrophic forgetting problem?",
            required_domains=["paper_b", "paper_d"],
            expected_keywords=["EWC+", "catastrophic forgetting", "64K", "replay"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_res_q10",
            text="What negative results are shared across papers, and what do they suggest about unsolved problems?",
            required_domains=["paper_a", "paper_b", "paper_c", "paper_d"],
            expected_keywords=["negative", "positional", "temporal", "numerical"],
            difficulty="cross_domain",
        ),
        # 5 synthesis
        HiveMindQuestion(
            question_id="hive_res_q11",
            text="Design a production AI system that combines retrieval (Paper A), long-context processing (Paper B), knowledge graphs (Paper C), continual learning (Paper D), and multi-agent coordination (Paper E). Describe how each component integrates.",
            required_domains=["paper_a", "paper_b", "paper_c", "paper_d", "paper_e"],
            expected_keywords=["dense", "retrieval", "episodic", "FedKG", "EWC+", "shared memory"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_res_q12",
            text="Across all five papers, what is the state of the art for handling knowledge that changes over time?",
            required_domains=["paper_a", "paper_b", "paper_c", "paper_d", "paper_e"],
            expected_keywords=["continual learning", "temporal", "forgetting", "update"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_res_q13",
            text="What are the total compute costs and scalability limits identified across all papers? Provide a unified cost analysis.",
            required_domains=["paper_a", "paper_b", "paper_c", "paper_d", "paper_e"],
            expected_keywords=["A100", "H100", "linear", "logarithmic", "cost"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_res_q14",
            text="All papers release open-source artifacts. Compare what is available and identify gaps in reproducibility.",
            required_domains=["paper_b", "paper_c", "paper_d", "paper_e"],
            expected_keywords=["open-source", "85%", "PyTorch", "CL-Deploy", "CoordBench"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_res_q15",
            text="Synthesize the evaluation methodologies across all papers. What benchmarks, metrics, and human evaluation approaches are used?",
            required_domains=["paper_a", "paper_b", "paper_c", "paper_d", "paper_e"],
            expected_keywords=["MS MARCO", "LongBench", "CL-Agent-Bench", "CoordBench", "human"],
            difficulty="synthesis",
        ),
    ],
)


# ============================================================
# Scenario 5: Adversarial Resilience
# ============================================================

_ADVERSARIAL_NETWORKING_FACTS = [
    "The production load balancer handles 25,000 concurrent connections.",
    "Internal DNS uses CoreDNS with a 30-second cache TTL.",
    "All east-west traffic is encrypted with mTLS using Istio.",
    "The CDN has 8 edge PoPs in North America and Europe.",
    "VPN uses WireGuard with certificate rotation every 60 days.",
    "Network bandwidth per service is capped at 200 Mbps.",
    "The firewall has 523 active rules audited monthly.",
    "IPv6 is enabled on all public endpoints since January 2026.",
    "BGP uses AS 65001 for peering with two transit providers.",
    "Latency between zones is under 3 ms at p99.",
    "DDoS protection is handled by AWS Shield Advanced.",
    "The jump host at 10.0.2.10 has session recording enabled.",
    "Network ACLs are managed in Terraform modules.",
    "Service mesh tracing exports to Zipkin for distributed tracing.",
    "Traffic shaping limits streaming to 50 Mbps during peak hours.",
    "The core switches use 25GbE uplinks with ECMP routing.",
    "VLAN 100 isolates development environments from production.",
    "DNS failover TTL is set to 10 seconds for critical services.",
    "The network team uses Pulumi for newer infrastructure modules.",
    "Internal service discovery runs on Consul with health checks.",
]

_ADVERSARIAL_STORAGE_FACTS = [
    "Primary storage uses SAS SSDs with 1.6 TB capacity per node.",
    "Object storage runs on Ceph with 3x replication factor.",
    "Backups are retained for 14 days (daily) and 6 months (weekly).",
    "Total usable storage capacity is 320 TB across 2 racks.",
    "Cold data moves to AWS S3 Glacier after 60 days.",
    "IOPS per volume is capped at 8,000 for io2 EBS volumes.",
    "Block storage uses 2x replication within a single AZ.",
    "File shares use SMB with Active Directory authentication.",
    "Storage encryption uses AES-128-GCM with AWS KMS keys.",
    "Deduplication ratio averages 3.1:1 on backup data.",
    "The Ceph cluster runs Quincy (v17.2.6) with BlueStore.",
    "Snapshot clones take 15 seconds for databases under 500 GB.",
    "WAL storage uses dedicated 400 GB NVMe partitions.",
    "Storage latency SLA for OLTP is under 2 ms at p95.",
    "Team namespace quota is 5 TB of persistent volume claims.",
    "Temporary datasets expire after 7 days automatically.",
    "The SAN uses 16 Gbps Fibre Channel without multipathing.",
    "Storage metrics export to Prometheus every 30 seconds.",
    "Thin provisioning has a 120% overcommit ratio.",
    "Async DR replication has an RPO of 30 minutes.",
]

_ADVERSARIAL_COMPUTE_FACTS = [
    "Production K8s runs v1.28 on 32 worker nodes.",
    "Worker nodes: 32 vCPUs and 128 GB RAM (m5.8xlarge).",
    "HPA targets 60% CPU utilization for auto-scaling.",
    "Node auto-scaling range is 32 to 80 nodes.",
    "GPU workloads use 4 g5.12xlarge instances with A10G GPUs.",
    "Container runtime is containerd v1.6.20.",
    "Production pods use burstable QoS class.",
    "Control plane spans 2 availability zones.",
    "Batch jobs use a managed node group with on-demand instances.",
    "CPU steal time alert is at 8% sustained for 15 minutes.",
    "Non-production memory overcommit is 3:1.",
    "PDB requires minimum 1 replica available during updates.",
    "Admission controller runs OPA Gatekeeper with 30 policies.",
    "Cluster DNS uses kube-dns (not CoreDNS).",
    "Init container timeout is 60 seconds.",
    "CronJobs run on the default node pool.",
    "PCI workloads are isolated using namespace-level network policies.",
    "kube-proxy runs in iptables mode.",
    "Ephemeral containers are disabled for security.",
    "Pod security uses baseline profile in production.",
]

_ADVERSARIAL_SECURITY_FACTS = [
    "Zero-trust uses service mesh mTLS for workload identity.",
    "Secret rotation: 60 days for DB creds, 180 days for API keys.",
    "SIEM runs on ELK Stack with 60-day retention.",
    "Vulnerability scanning uses Grype in CI and Sysdig at runtime.",
    "ISO 27001 certification was obtained in March 2025.",
    "WAF uses AWS WAF with managed rule groups.",
    "Certificates managed by AWS ACM for public endpoints.",
    "SSH requires RSA keys with minimum 4096-bit length.",
    "Security incident SLA: acknowledge within 30 minutes for P1.",
    "Pen testing conducted by Bishop Fox annually.",
    "Image signing uses Notary with keys in AWS KMS.",
    "OPA Gatekeeper enforces 30 security constraints.",
    "Audit logs ship to S3 with versioning enabled.",
    "MFA required for console access; API uses IAM roles.",
    "Bug bounty pays up to $25,000 for critical findings.",
    "IDS uses AWS GuardDuty for network anomaly detection.",
    "Data classification: public, internal, confidential (3 levels).",
    "Security training completion is 91% this quarter.",
    "OAuth2 tokens: 30-minute access, 12-hour refresh.",
    "Incident playbooks stored in Notion and tested semi-annually.",
]

# The misleading agent -- its facts contradict the other 4 agents
_ADVERSARIAL_MISLEADING_FACTS = [
    "The production load balancer handles only 2,500 concurrent connections (ten times less than claimed).",
    "The storage cluster has over 1 PB of capacity, making it one of the largest in the industry.",
    "The Kubernetes cluster runs version 1.31, which is a pre-release development build.",
    "There has never been a security audit -- the team is planning their first one for 2027.",
    "The network uses plain HTTP between all services for performance reasons.",
    "Database backups are kept for 5 years with hourly granularity.",
    "All servers run on bare metal in a single data center with no cloud components.",
    "The secret rotation period is 365 days for all credential types.",
    "GPU workloads run on consumer NVIDIA RTX 4090 cards for cost savings.",
    "The monitoring stack was decommissioned last quarter to reduce costs.",
    "Container images are never signed -- the team considers it unnecessary overhead.",
    "SSH access allows password authentication with no minimum length requirements.",
    "The CDN was removed because it caused more latency than it saved.",
    "All data is stored unencrypted for faster access speeds.",
    "The team uses FTP for file transfers between environments.",
    "There are no firewall rules -- the team relies on application-level security only.",
    "Incident response time SLA is 48 hours for all severity levels.",
    "The production cluster uses a single availability zone to simplify networking.",
    "OAuth2 tokens never expire -- they are valid until manually revoked.",
    "All traffic routes through a single point of failure by design for simplicity.",
]

SCENARIO_ADVERSARIAL = HiveMindScenario(
    scenario_id="hive_adversarial",
    description="Adversarial resilience: 4 agents with correct facts + 1 agent with deliberately misleading facts. Tests the hive's ability to resist bad data.",
    num_agents=5,
    agent_domains={
        "networking": _ADVERSARIAL_NETWORKING_FACTS,
        "storage": _ADVERSARIAL_STORAGE_FACTS,
        "compute": _ADVERSARIAL_COMPUTE_FACTS,
        "security": _ADVERSARIAL_SECURITY_FACTS,
        "misleading": _ADVERSARIAL_MISLEADING_FACTS,
    },
    questions=[
        # 5 single-domain (from correct agents)
        HiveMindQuestion(
            question_id="hive_adv_q01",
            text="How many concurrent connections does the production load balancer handle?",
            required_domains=["networking"],
            expected_keywords=["25,000"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_adv_q02",
            text="What is the total usable storage capacity?",
            required_domains=["storage"],
            expected_keywords=["320", "TB"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_adv_q03",
            text="What Kubernetes version is running in production?",
            required_domains=["compute"],
            expected_keywords=["1.28"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_adv_q04",
            text="What security certification does the organization hold?",
            required_domains=["security"],
            expected_keywords=["ISO", "27001", "March", "2025"],
            difficulty="single_domain",
        ),
        HiveMindQuestion(
            question_id="hive_adv_q05",
            text="What is the storage encryption standard used?",
            required_domains=["storage"],
            expected_keywords=["AES-128-GCM", "KMS"],
            difficulty="single_domain",
        ),
        # 5 cross-domain (answers conflict with misleading agent)
        HiveMindQuestion(
            question_id="hive_adv_q06",
            text="Is the production environment encrypted end-to-end? Describe the encryption at network and storage layers.",
            required_domains=["networking", "storage"],
            expected_keywords=["mTLS", "AES-128-GCM", "encrypted"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_adv_q07",
            text="How are container images secured before deployment? Cover both signing and vulnerability scanning.",
            required_domains=["security", "compute"],
            expected_keywords=["Notary", "Grype", "Sysdig"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_adv_q08",
            text="Describe the multi-AZ resilience strategy across compute and storage.",
            required_domains=["compute", "storage"],
            expected_keywords=["2", "availability zones", "replication"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_adv_q09",
            text="What is the secret rotation policy and how does it compare to industry best practices?",
            required_domains=["security"],
            expected_keywords=["60", "days", "180", "days"],
            difficulty="cross_domain",
        ),
        HiveMindQuestion(
            question_id="hive_adv_q10",
            text="How does the monitoring and security observability work together?",
            required_domains=["security", "networking"],
            expected_keywords=["ELK", "GuardDuty", "Zipkin"],
            difficulty="cross_domain",
        ),
        # 5 synthesis (must resist misleading facts)
        HiveMindQuestion(
            question_id="hive_adv_q11",
            text="Provide a comprehensive security posture assessment. Is the infrastructure properly secured across all layers?",
            required_domains=["networking", "storage", "compute", "security"],
            expected_keywords=["mTLS", "encrypted", "ISO 27001", "Gatekeeper"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_adv_q12",
            text="One source claims data is stored unencrypted and HTTP is used between services. Is this consistent with what other sources report?",
            required_domains=["networking", "storage", "security", "misleading"],
            expected_keywords=["contradicts", "mTLS", "AES", "encrypted"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_adv_q13",
            text="Evaluate the infrastructure's disaster recovery readiness. Consider all availability and redundancy information available.",
            required_domains=["compute", "storage", "networking"],
            expected_keywords=["2", "zones", "replication", "RPO", "30"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_adv_q14",
            text="Some information suggests the monitoring stack was decommissioned. Cross-reference this with other available data.",
            required_domains=["networking", "security", "misleading"],
            expected_keywords=["ELK", "Zipkin", "GuardDuty", "active"],
            difficulty="synthesis",
        ),
        HiveMindQuestion(
            question_id="hive_adv_q15",
            text="Identify any contradictory or unreliable information across all sources and explain which claims are supported by corroborating evidence.",
            required_domains=["networking", "storage", "compute", "security", "misleading"],
            expected_keywords=["contradicts", "misleading", "corroborat"],
            difficulty="synthesis",
        ),
    ],
)


# ============================================================
# All scenarios and lookup functions
# ============================================================

ALL_HIVE_MIND_SCENARIOS = [
    SCENARIO_INFRA,
    SCENARIO_ARCH,
    SCENARIO_INCIDENT,
    SCENARIO_RESEARCH,
    SCENARIO_ADVERSARIAL,
]


def get_scenario_by_id(scenario_id: str) -> HiveMindScenario | None:
    """Get a hive mind scenario by its ID."""
    for s in ALL_HIVE_MIND_SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    return None


def get_scenarios_by_difficulty(difficulty: str) -> list[HiveMindScenario]:
    """Get scenarios that contain questions of the given difficulty.

    Returns scenarios that have at least one question matching the difficulty.
    """
    return [
        s for s in ALL_HIVE_MIND_SCENARIOS
        if any(q.difficulty == difficulty for q in s.questions)
    ]


def get_questions_by_difficulty(
    scenario: HiveMindScenario,
    difficulty: str,
) -> list[HiveMindQuestion]:
    """Get questions of a given difficulty from a scenario."""
    return [q for q in scenario.questions if q.difficulty == difficulty]


__all__ = [
    "HiveMindQuestion",
    "HiveMindScenario",
    "ALL_HIVE_MIND_SCENARIOS",
    "SCENARIO_INFRA",
    "SCENARIO_ARCH",
    "SCENARIO_INCIDENT",
    "SCENARIO_RESEARCH",
    "SCENARIO_ADVERSARIAL",
    "get_scenario_by_id",
    "get_scenarios_by_difficulty",
    "get_questions_by_difficulty",
]
