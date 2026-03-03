"""Cloud security analyst scenario with unified L1-L12 progressive levels.

Philosophy:
- Deterministic data generation -- all content is template-based, no LLM needed
- Reproducible: same seed produces identical dialogue every time
- Ground truth tracked for every fact delivered
- L1-L12 levels map to cognitive complexity, not categories
- Works with BOTH EvalRunner.run() (single agent) and hive mind eval scripts
- Reuses Turn, Question, GradingRubric, GroundTruth from long_horizon.py

Public API:
    generate_dialogue(num_turns, seed) -> GroundTruth
    generate_questions(ground_truth, num_questions) -> list[Question]
    LEVEL_NAMES: dict mapping L1-L12 to human-readable names
"""

from __future__ import annotations

import logging
import random
from typing import Any

from .long_horizon import GradingRubric, GroundTruth, Question, Turn

logger = logging.getLogger(__name__)

# ============================================================
# L1-L12 Level Definitions
# ============================================================

LEVEL_NAMES: dict[str, str] = {
    "L1": "Direct Recall",
    "L2": "Multi-Source Synthesis",
    "L3": "Temporal Reasoning",
    "L4": "Procedural Knowledge",
    "L5": "Contradiction Resolution",
    "L6": "Incremental Update",
    "L7": "Teaching / Explanation",
    "L8": "Confidence Calibration",
    "L9": "Causal Reasoning",
    "L10": "Counterfactual Reasoning",
    "L11": "Novel Skill Application",
    "L12": "Far Transfer",
}

# Percentage allocation for each level when generating questions
LEVEL_DISTRIBUTION: dict[str, float] = {
    "L1": 0.12,
    "L2": 0.10,
    "L3": 0.10,
    "L4": 0.08,
    "L5": 0.08,
    "L6": 0.08,
    "L7": 0.08,
    "L8": 0.08,
    "L9": 0.08,
    "L10": 0.06,
    "L11": 0.06,
    "L12": 0.08,
}

# ============================================================
# Infrastructure Data
# ============================================================

SERVERS = [
    {
        "name": "prod-db-01",
        "ip": "10.0.1.10",
        "service": "PostgreSQL 15.4",
        "port": 5432,
        "ram_gb": 64,
        "cpu_cores": 16,
        "os": "Ubuntu 22.04 LTS",
        "subnet": "production",
        "role": "Primary database",
        "disk_gb": 2000,
        "last_patched": "2024-01-10",
    },
    {
        "name": "prod-web-01",
        "ip": "10.0.1.20",
        "service": "Nginx 1.25.3",
        "port": 443,
        "ram_gb": 32,
        "cpu_cores": 8,
        "os": "Ubuntu 22.04 LTS",
        "subnet": "production",
        "role": "Web frontend / TLS termination",
        "disk_gb": 500,
        "last_patched": "2024-01-12",
    },
    {
        "name": "prod-app-01",
        "ip": "10.0.1.30",
        "service": "Java/Spring Boot 3.2",
        "port": 8443,
        "ram_gb": 32,
        "cpu_cores": 8,
        "os": "Ubuntu 22.04 LTS",
        "subnet": "production",
        "role": "Application server",
        "disk_gb": 500,
        "last_patched": "2024-01-08",
    },
    {
        "name": "prod-cache-01",
        "ip": "10.0.1.40",
        "service": "Redis 7.2",
        "port": 6379,
        "ram_gb": 16,
        "cpu_cores": 4,
        "os": "Ubuntu 22.04 LTS",
        "subnet": "production",
        "role": "Session cache and rate limiter",
        "disk_gb": 200,
        "last_patched": "2024-01-14",
    },
    {
        "name": "staging-db-01",
        "ip": "10.0.2.10",
        "service": "PostgreSQL 15.4",
        "port": 5432,
        "ram_gb": 16,
        "cpu_cores": 4,
        "os": "Ubuntu 22.04 LTS",
        "subnet": "staging",
        "role": "Staging database",
        "disk_gb": 500,
        "last_patched": "2024-01-05",
    },
    {
        "name": "staging-app-01",
        "ip": "10.0.2.20",
        "service": "Java/Spring Boot 3.2",
        "port": 8443,
        "ram_gb": 16,
        "cpu_cores": 4,
        "os": "Ubuntu 22.04 LTS",
        "subnet": "staging",
        "role": "Staging application server",
        "disk_gb": 500,
        "last_patched": "2024-01-05",
    },
    {
        "name": "monitoring-01",
        "ip": "10.0.3.10",
        "service": "Prometheus 2.48 / Grafana 10.2",
        "port": 9090,
        "ram_gb": 32,
        "cpu_cores": 8,
        "os": "Ubuntu 22.04 LTS",
        "subnet": "management",
        "role": "Monitoring and alerting",
        "disk_gb": 1000,
        "last_patched": "2024-01-15",
    },
    {
        "name": "bastion-01",
        "ip": "10.0.0.5",
        "service": "OpenSSH 9.5",
        "port": 22,
        "ram_gb": 4,
        "cpu_cores": 2,
        "os": "Ubuntu 22.04 LTS",
        "subnet": "dmz",
        "role": "SSH bastion / jump host",
        "disk_gb": 100,
        "last_patched": "2024-01-15",
    },
]

NETWORK_TOPOLOGY = {
    "subnets": {
        "dmz": {"cidr": "10.0.0.0/24", "purpose": "Internet-facing services and bastion"},
        "production": {"cidr": "10.0.1.0/24", "purpose": "Production workloads"},
        "staging": {"cidr": "10.0.2.0/24", "purpose": "Staging and pre-prod testing"},
        "management": {"cidr": "10.0.3.0/24", "purpose": "Monitoring, logging, admin tools"},
    },
    "load_balancers": [
        {
            "name": "alb-prod-web",
            "type": "ALB",
            "ip": "10.0.0.100",
            "targets": ["prod-web-01"],
            "port": 443,
        },
        {
            "name": "nlb-prod-api",
            "type": "NLB",
            "ip": "10.0.0.101",
            "targets": ["prod-app-01"],
            "port": 8443,
        },
    ],
    "vpn_gateway": "10.0.0.1",
}

# ============================================================
# Security Incidents
# ============================================================

INCIDENTS = [
    {
        "id": "INC-2024-001",
        "title": "Ransomware attempt via phishing",
        "severity": "critical",
        "detected": "2024-01-15 03:42 UTC",
        "resolved": "2024-01-15 14:30 UTC",
        "cause": "Employee clicked phishing link delivering Lockbit payload",
        "affected_servers": ["prod-app-01"],
        "impact": "Application server isolated, 2 hours downtime",
        "root_cause": "Spear-phishing email bypassed email gateway; endpoint detection caught payload before encryption started",
        "timeline": [
            "03:42 - EDR alert on prod-app-01 for suspicious process execution",
            "03:45 - SOC analyst Bob Chen acknowledged alert",
            "04:00 - Server isolated from network via firewall rule",
            "06:00 - Forensic image captured",
            "10:00 - Malware identified as Lockbit variant (hash: a1b2c3d4)",
            "12:00 - Clean rebuild from golden image initiated",
            "14:30 - Server restored and verified clean",
        ],
        "remediation": "Email gateway rules updated, mandatory phishing training scheduled, EDR policies tightened",
    },
    {
        "id": "INC-2024-002",
        "title": "API key exfiltration via compromised CI/CD",
        "severity": "high",
        "detected": "2024-02-03 11:15 UTC",
        "resolved": "2024-02-04 09:00 UTC",
        "cause": "Hardcoded API key in CI pipeline config exposed via public GitHub Actions log",
        "affected_servers": ["prod-app-01", "prod-db-01"],
        "impact": "service-payment API key compromised; 47 unauthorized API calls detected",
        "root_cause": "Developer committed .env file with production API key to public fork; CI log exposed key in debug output",
        "timeline": [
            "11:15 - Anomaly detection flagged 47 API calls from unknown IP 203.0.113.50",
            "11:30 - Alice Kim confirmed key compromise",
            "11:45 - service-payment key revoked and rotated",
            "12:00 - IP 203.0.113.50 blocked at WAF",
            "15:00 - Audit of all API call logs from compromised key completed",
            "2024-02-04 09:00 - Incident closed after full remediation",
        ],
        "remediation": "Secret scanning enabled on all repos, API keys moved to vault, CI debug logging restricted",
    },
    {
        "id": "INC-2024-003",
        "title": "DDoS attack on web frontend",
        "severity": "medium",
        "detected": "2024-03-10 16:00 UTC",
        "resolved": "2024-03-10 18:30 UTC",
        "cause": "Volumetric DDoS (UDP flood) targeting prod-web-01",
        "affected_servers": ["prod-web-01"],
        "impact": "Intermittent 503 errors for 45 minutes, peak 2.5 Gbps traffic",
        "root_cause": "Booter service targeting public IP; no specific motivation identified",
        "timeline": [
            "16:00 - Monitoring alert: prod-web-01 CPU at 95%, response time >5s",
            "16:15 - Carol Davis identified UDP flood pattern from traffic analysis",
            "16:30 - ISP upstream filtering activated",
            "17:00 - Rate limiting policy v2 deployed to Nginx",
            "18:00 - Traffic normalized",
            "18:30 - Incident closed",
        ],
        "remediation": "Rate limiting added (100 req/s per IP), DDoS mitigation service contract signed, runbook updated",
    },
]

# ============================================================
# CVE Advisories
# ============================================================

CVES = [
    {
        "id": "CVE-2024-1234",
        "title": "OpenSSL buffer overflow in TLS handshake",
        "cvss": 9.1,
        "severity": "critical",
        "published": "2024-01-05",
        "affected_software": "OpenSSL 3.0.x < 3.0.13",
        "affected_servers": ["prod-web-01", "prod-app-01"],
        "description": "A buffer overflow in the TLS 1.3 handshake allows remote code execution",
        "patch_status": {
            "prod-web-01": "patched 2024-01-12",
            "prod-app-01": "patched 2024-01-08",
        },
        "remediation": "Upgrade OpenSSL to 3.0.13 or later",
    },
    {
        "id": "CVE-2024-5678",
        "title": "Linux kernel privilege escalation via netfilter",
        "cvss": 7.8,
        "severity": "high",
        "published": "2024-02-20",
        "affected_software": "Linux kernel 5.15.x - 6.5.x",
        "affected_servers": [
            "prod-db-01",
            "prod-web-01",
            "prod-app-01",
            "prod-cache-01",
            "staging-db-01",
            "staging-app-01",
            "monitoring-01",
            "bastion-01",
        ],
        "description": "Local privilege escalation via crafted netfilter rules allows root access",
        "patch_status": {
            "prod-db-01": "patched 2024-02-25",
            "prod-web-01": "patched 2024-02-25",
            "prod-app-01": "patched 2024-02-25",
            "prod-cache-01": "patched 2024-02-25",
            "staging-db-01": "unpatched",
            "staging-app-01": "unpatched",
            "monitoring-01": "patched 2024-02-26",
            "bastion-01": "patched 2024-02-25",
        },
        "remediation": "Apply kernel update to 6.5.13 or later",
    },
    {
        "id": "CVE-2024-9012",
        "title": "PostgreSQL SQL injection in pg_stat_statements",
        "cvss": 8.4,
        "severity": "high",
        "published": "2024-03-01",
        "affected_software": "PostgreSQL 15.0 - 15.5",
        "affected_servers": ["prod-db-01", "staging-db-01"],
        "description": "SQL injection via crafted query to pg_stat_statements extension allows data exfiltration",
        "patch_status": {
            "prod-db-01": "patched 2024-03-05",
            "staging-db-01": "unpatched",
        },
        "remediation": "Upgrade PostgreSQL to 15.6 or disable pg_stat_statements",
    },
]

# ============================================================
# Vulnerability Scan Results
# ============================================================

SCAN_RESULTS = [
    {
        "scan_date": "2024-03-15",
        "scanner": "Nessus 10.6",
        "scope": "all production and staging servers",
        "findings": [
            {
                "server": "prod-web-01",
                "finding": "TLS 1.0 still enabled",
                "severity": "medium",
                "status": "open",
            },
            {
                "server": "prod-app-01",
                "finding": "Java 17.0.6 has known CVE (CVE-2024-1234 mitigated at OS level)",
                "severity": "low",
                "status": "accepted risk",
            },
            {
                "server": "staging-db-01",
                "finding": "PostgreSQL 15.4 unpatched for CVE-2024-9012",
                "severity": "high",
                "status": "open",
            },
            {
                "server": "staging-app-01",
                "finding": "Linux kernel unpatched for CVE-2024-5678",
                "severity": "high",
                "status": "open",
            },
            {
                "server": "bastion-01",
                "finding": "SSH allows password authentication (should be key-only)",
                "severity": "medium",
                "status": "open",
            },
        ],
    },
]

# ============================================================
# Team and On-Call Schedule
# ============================================================

TEAM = [
    {
        "name": "Alice Kim",
        "role": "Security Lead",
        "email": "alice.kim@example.com",
        "on_call": "Monday-Wednesday",
        "specialties": ["incident response", "threat hunting", "compliance"],
        "certifications": ["CISSP", "OSCP"],
    },
    {
        "name": "Bob Chen",
        "role": "SOC Analyst",
        "email": "bob.chen@example.com",
        "on_call": "Thursday-Friday",
        "specialties": ["SIEM monitoring", "alert triage", "log analysis"],
        "certifications": ["Security+", "CySA+"],
    },
    {
        "name": "Carol Davis",
        "role": "Incident Response Engineer",
        "email": "carol.davis@example.com",
        "on_call": "Saturday-Sunday",
        "specialties": ["forensics", "malware analysis", "containment"],
        "certifications": ["GCIH", "GCFA"],
    },
    {
        "name": "Dave Park",
        "role": "Compliance Officer",
        "email": "dave.park@example.com",
        "on_call": "N/A (business hours only)",
        "specialties": ["SOC 2", "PCI-DSS", "audit management"],
        "certifications": ["CISA", "CRISC"],
    },
]

# ============================================================
# Firewall Policy Versions
# ============================================================

FIREWALL_POLICIES = [
    {
        "version": "v1",
        "date": "2024-01-01",
        "description": "Baseline firewall policy",
        "rules": [
            "ALLOW TCP 443 inbound from 0.0.0.0/0 to 10.0.0.0/24 (HTTPS)",
            "ALLOW TCP 22 inbound from VPN (10.10.0.0/16) to bastion-01",
            "ALLOW ALL from production subnet to production subnet",
            "DENY ALL inbound from internet to production/staging/management subnets",
            "ALLOW TCP 5432 from prod-app-01 to prod-db-01",
            "ALLOW TCP 6379 from prod-app-01 to prod-cache-01",
        ],
    },
    {
        "version": "v2",
        "date": "2024-03-10",
        "description": "Post-DDoS mitigation update",
        "changes": [
            "ADDED rate limiting: 100 requests/second per source IP on port 443",
            "ADDED UDP flood protection: drop UDP packets > 1000/s from single source",
            "ADDED geo-blocking for known botnet source countries",
        ],
    },
    {
        "version": "v3",
        "date": "2024-03-20",
        "description": "Post-exfiltration hardening",
        "changes": [
            "BLOCKED outbound to 47 known C2 IP addresses",
            "ADDED egress filtering: only allow outbound to approved IP ranges",
            "ADDED TLS inspection on outbound HTTPS from production subnet",
            "RESTRICTED staging subnet: no outbound internet access",
        ],
    },
]

# ============================================================
# API Key Rotation Records
# ============================================================

API_KEYS = [
    {
        "service": "service-payment",
        "key_id": "sk_pay_v3",
        "created": "2024-02-03",
        "expires": "2024-08-03",
        "rotation_reason": "Compromised during INC-2024-002",
        "previous_key_id": "sk_pay_v2",
        "rotation_procedure": [
            "1. Generate new key in payment provider dashboard",
            "2. Update key in HashiCorp Vault at secret/prod/payment-api-key",
            "3. Trigger rolling restart of prod-app-01 to pick up new key",
            "4. Verify payment processing with test transaction",
            "5. Revoke old key in payment provider dashboard",
            "6. Update rotation log in Confluence",
        ],
    },
    {
        "service": "service-auth",
        "key_id": "sk_auth_v5",
        "created": "2024-02-01",
        "expires": "2024-08-01",
        "rotation_reason": "Scheduled 180-day rotation",
        "previous_key_id": "sk_auth_v4",
        "rotation_procedure": [
            "1. Generate new key pair using auth service CLI",
            "2. Update public key in identity provider",
            "3. Update private key in Vault at secret/prod/auth-signing-key",
            "4. Deploy auth service with new key (blue-green deployment)",
            "5. Verify token validation with integration tests",
            "6. Deactivate old key after 24-hour grace period",
        ],
    },
    {
        "service": "service-monitoring",
        "key_id": "sk_mon_v2",
        "created": "2024-01-15",
        "expires": "2024-07-15",
        "rotation_reason": "Scheduled 180-day rotation",
        "previous_key_id": "sk_mon_v1",
        "rotation_procedure": [
            "1. Generate new API token in Prometheus/Grafana",
            "2. Update token in Vault at secret/prod/monitoring-api-key",
            "3. Restart monitoring agents on all servers",
            "4. Verify metrics collection resumes within 5 minutes",
            "5. Revoke old token",
        ],
    },
]

# ============================================================
# Audit Logs
# ============================================================

AUDIT_LOGS = [
    {
        "timestamp": "2024-01-15 03:40 UTC",
        "actor": "user:jsmith@example.com",
        "action": "login",
        "resource": "prod-app-01",
        "details": "SSH login from 10.10.5.22 (VPN)",
        "result": "success",
    },
    {
        "timestamp": "2024-01-15 03:41 UTC",
        "actor": "user:jsmith@example.com",
        "action": "download",
        "resource": "prod-app-01:/tmp/invoice.pdf",
        "details": "Downloaded suspicious file from phishing link",
        "result": "success",
    },
    {
        "timestamp": "2024-01-15 03:42 UTC",
        "actor": "process:lockbit.exe",
        "action": "execute",
        "resource": "prod-app-01",
        "details": "Ransomware payload executed, caught by EDR before encryption",
        "result": "blocked by EDR",
    },
    {
        "timestamp": "2024-02-03 10:00 UTC",
        "actor": "ci:github-actions",
        "action": "deploy",
        "resource": "prod-app-01",
        "details": "Deploy build #1847 with debug logging enabled",
        "result": "success",
    },
    {
        "timestamp": "2024-02-03 11:10 UTC",
        "actor": "external:203.0.113.50",
        "action": "api_call",
        "resource": "service-payment",
        "details": "47 unauthorized calls using compromised API key sk_pay_v2",
        "result": "success (key valid at time)",
    },
    {
        "timestamp": "2024-03-10 16:00 UTC",
        "actor": "external:multiple",
        "action": "flood",
        "resource": "prod-web-01",
        "details": "UDP flood from ~5000 source IPs, peak 2.5 Gbps",
        "result": "partial degradation",
    },
    {
        "timestamp": "2024-03-15 09:00 UTC",
        "actor": "scanner:nessus",
        "action": "vulnerability_scan",
        "resource": "all servers",
        "details": "Quarterly vulnerability scan completed, 5 findings",
        "result": "completed",
    },
]

# ============================================================
# Deployment History
# ============================================================

DEPLOYMENTS = [
    {
        "version": "v2.3",
        "date": "2024-01-05",
        "server": "prod-app-01",
        "changes": [
            "Added OAuth 2.0 PKCE flow for mobile clients",
            "Fixed session timeout bug (sessions now expire after 30 minutes idle)",
            "Updated Spring Boot to 3.2.1",
        ],
    },
    {
        "version": "v2.4",
        "date": "2024-03-12",
        "server": "prod-app-01",
        "changes": [
            "Added rate limiting middleware (100 req/s per IP, using Redis)",
            "Fixed session cookie SameSite attribute (was Lax, now Strict)",
            "Added request ID header for distributed tracing",
            "Upgraded Jackson JSON library to 2.16.1 (security patch)",
        ],
    },
]

# ============================================================
# Policy Documents
# ============================================================

POLICIES = [
    {
        "name": "Incident Response Procedure",
        "version": "3.1",
        "last_updated": "2024-03-15",
        "steps": [
            "1. DETECT: Acknowledge alert within 15 minutes of notification",
            "2. TRIAGE: Classify severity (critical/high/medium/low) within 30 minutes",
            "3. CONTAIN: Isolate affected systems — network segmentation or shutdown",
            "4. INVESTIGATE: Capture forensic evidence, identify root cause",
            "5. ERADICATE: Remove threat actor access, patch vulnerabilities",
            "6. RECOVER: Restore from clean backups, verify integrity",
            "7. LESSONS LEARNED: Post-incident review within 5 business days",
        ],
        "escalation": "Critical: page Security Lead immediately. High: notify within 1 hour. Medium/Low: next business day.",
    },
    {
        "name": "Patch Management Policy",
        "version": "2.0",
        "last_updated": "2024-02-28",
        "rules": [
            "Critical CVEs (CVSS >= 9.0): patch within 48 hours",
            "High CVEs (CVSS 7.0-8.9): patch within 7 days",
            "Medium CVEs (CVSS 4.0-6.9): patch within 30 days",
            "Low CVEs (CVSS < 4.0): patch in next maintenance window",
            "All patches must be tested in staging before production deployment",
            "Emergency patches may skip staging with Security Lead approval",
        ],
    },
]

# ============================================================
# Contradiction data (for L5 questions)
# ============================================================

CONTRADICTIONS = [
    {
        "topic": "prod-app-01 patch status for CVE-2024-1234",
        "source_a": {
            "source": "INC-2024-001 post-mortem (2024-01-16)",
            "claim": "prod-app-01 was rebuilt from golden image on 2024-01-15, which includes latest patches including CVE-2024-1234 fix",
        },
        "source_b": {
            "source": "Nessus scan (2024-03-15)",
            "claim": "prod-app-01 Java runtime 17.0.6 has known CVE-2024-1234 mitigated at OS level — the Java runtime itself is not updated",
        },
        "resolution": "Both are partially correct: the OS-level OpenSSL was patched (golden image), but the Java runtime bundled its own OpenSSL that remains at the old version. The risk is accepted because TLS terminates at Nginx, not at the Java layer.",
    },
]

# ============================================================
# Novel API documentation (for L11 questions)
# ============================================================

NOVEL_API_DOCS = {
    "provider": "CloudWatch Pro (fictional)",
    "version": "API v3",
    "monitoring_config_format": {
        "type": "YAML",
        "example": """monitors:
  - name: "cpu-alert"
    metric: "system.cpu.percent"
    threshold: 85
    duration: "5m"
    severity: "warning"
    notify: ["slack:#ops-alerts"]
  - name: "disk-alert"
    metric: "system.disk.used_percent"
    threshold: 90
    duration: "10m"
    severity: "critical"
    notify: ["pagerduty:security-oncall"]""",
        "required_fields": ["name", "metric", "threshold", "duration", "severity", "notify"],
    },
}


# ============================================================
# Dialogue Generation
# ============================================================


def _make_server_turn(server: dict[str, Any], turn_number: int) -> Turn:
    """Generate a turn about a server configuration."""
    content = (
        f"Server Configuration: {server['name']}\n"
        f"IP Address: {server['ip']}\n"
        f"Service: {server['service']} on port {server['port']}\n"
        f"RAM: {server['ram_gb']}GB, CPU: {server['cpu_cores']} cores\n"
        f"OS: {server['os']}\n"
        f"Subnet: {server['subnet']} ({NETWORK_TOPOLOGY['subnets'][server['subnet']]['cidr']})\n"
        f"Role: {server['role']}\n"
        f"Storage: {server['disk_gb']}GB\n"
        f"Last Patched: {server['last_patched']}"
    )
    facts = [
        {"entity": server["name"], "field": "ip", "value": server["ip"]},
        {"entity": server["name"], "field": "service", "value": server["service"]},
        {"entity": server["name"], "field": "port", "value": str(server["port"])},
        {"entity": server["name"], "field": "ram_gb", "value": str(server["ram_gb"])},
        {"entity": server["name"], "field": "subnet", "value": server["subnet"]},
        {"entity": server["name"], "field": "role", "value": server["role"]},
        {"entity": server["name"], "field": "os", "value": server["os"]},
        {"entity": server["name"], "field": "last_patched", "value": server["last_patched"]},
    ]
    return Turn(
        turn_number=turn_number,
        content=content,
        block=1,
        block_name="infrastructure",
        facts=facts,
    )


def _make_incident_turn(incident: dict[str, Any], turn_number: int) -> Turn:
    """Generate a turn about a security incident."""
    timeline_str = "\n".join(f"  {t}" for t in incident["timeline"])
    content = (
        f"Security Incident Report: {incident['id']}\n"
        f"Title: {incident['title']}\n"
        f"Severity: {incident['severity']}\n"
        f"Detected: {incident['detected']}\n"
        f"Resolved: {incident['resolved']}\n"
        f"Cause: {incident['cause']}\n"
        f"Affected Servers: {', '.join(incident['affected_servers'])}\n"
        f"Impact: {incident['impact']}\n"
        f"Root Cause: {incident['root_cause']}\n"
        f"Timeline:\n{timeline_str}\n"
        f"Remediation: {incident['remediation']}"
    )
    facts = [
        {"entity": incident["id"], "field": "title", "value": incident["title"]},
        {"entity": incident["id"], "field": "severity", "value": incident["severity"]},
        {"entity": incident["id"], "field": "detected", "value": incident["detected"]},
        {"entity": incident["id"], "field": "resolved", "value": incident["resolved"]},
        {"entity": incident["id"], "field": "cause", "value": incident["cause"]},
        {
            "entity": incident["id"],
            "field": "affected_servers",
            "value": ", ".join(incident["affected_servers"]),
        },
        {"entity": incident["id"], "field": "impact", "value": incident["impact"]},
        {"entity": incident["id"], "field": "root_cause", "value": incident["root_cause"]},
        {"entity": incident["id"], "field": "remediation", "value": incident["remediation"]},
    ]
    return Turn(
        turn_number=turn_number,
        content=content,
        block=2,
        block_name="incidents",
        facts=facts,
    )


def _make_cve_turn(cve: dict[str, Any], turn_number: int) -> Turn:
    """Generate a turn about a CVE advisory."""
    patch_lines = "\n".join(
        f"  {server}: {status}" for server, status in cve["patch_status"].items()
    )
    content = (
        f"CVE Advisory: {cve['id']}\n"
        f"Title: {cve['title']}\n"
        f"CVSS Score: {cve['cvss']}\n"
        f"Severity: {cve['severity']}\n"
        f"Published: {cve['published']}\n"
        f"Affected Software: {cve['affected_software']}\n"
        f"Affected Servers: {', '.join(cve['affected_servers'])}\n"
        f"Description: {cve['description']}\n"
        f"Patch Status:\n{patch_lines}\n"
        f"Remediation: {cve['remediation']}"
    )
    facts = [
        {"entity": cve["id"], "field": "title", "value": cve["title"]},
        {"entity": cve["id"], "field": "cvss", "value": str(cve["cvss"])},
        {"entity": cve["id"], "field": "severity", "value": cve["severity"]},
        {
            "entity": cve["id"],
            "field": "affected_servers",
            "value": ", ".join(cve["affected_servers"]),
        },
        {"entity": cve["id"], "field": "description", "value": cve["description"]},
        {"entity": cve["id"], "field": "remediation", "value": cve["remediation"]},
    ]
    for server, status in cve["patch_status"].items():
        facts.append({"entity": cve["id"], "field": f"patch_{server}", "value": status})
    return Turn(
        turn_number=turn_number,
        content=content,
        block=3,
        block_name="vulnerabilities",
        facts=facts,
    )


def _make_team_turn(member: dict[str, Any], turn_number: int) -> Turn:
    """Generate a turn about a team member."""
    content = (
        f"Team Member: {member['name']}\n"
        f"Role: {member['role']}\n"
        f"Email: {member['email']}\n"
        f"On-Call Schedule: {member['on_call']}\n"
        f"Specialties: {', '.join(member['specialties'])}\n"
        f"Certifications: {', '.join(member['certifications'])}"
    )
    facts = [
        {"entity": member["name"], "field": "role", "value": member["role"]},
        {"entity": member["name"], "field": "on_call", "value": member["on_call"]},
        {
            "entity": member["name"],
            "field": "specialties",
            "value": ", ".join(member["specialties"]),
        },
        {
            "entity": member["name"],
            "field": "certifications",
            "value": ", ".join(member["certifications"]),
        },
    ]
    return Turn(
        turn_number=turn_number,
        content=content,
        block=4,
        block_name="team",
        facts=facts,
    )


def _make_firewall_turn(policy: dict[str, Any], turn_number: int) -> Turn:
    """Generate a turn about a firewall policy version."""
    if "rules" in policy:
        rules_str = "\n".join(f"  {r}" for r in policy["rules"])
        content = (
            f"Firewall Policy {policy['version']} ({policy['date']})\n"
            f"Description: {policy['description']}\n"
            f"Rules:\n{rules_str}"
        )
    else:
        changes_str = "\n".join(f"  {c}" for c in policy["changes"])
        content = (
            f"Firewall Policy Update {policy['version']} ({policy['date']})\n"
            f"Description: {policy['description']}\n"
            f"Changes from previous version:\n{changes_str}"
        )
    facts = [
        {
            "entity": f"firewall_{policy['version']}",
            "field": "date",
            "value": policy["date"],
        },
        {
            "entity": f"firewall_{policy['version']}",
            "field": "description",
            "value": policy["description"],
        },
    ]
    return Turn(
        turn_number=turn_number,
        content=content,
        block=5,
        block_name="policies",
        facts=facts,
    )


def _make_api_key_turn(key: dict[str, Any], turn_number: int) -> Turn:
    """Generate a turn about API key rotation."""
    procedure_str = "\n".join(f"  {s}" for s in key["rotation_procedure"])
    content = (
        f"API Key Rotation Record: {key['service']}\n"
        f"Current Key ID: {key['key_id']}\n"
        f"Created: {key['created']}\n"
        f"Expires: {key['expires']}\n"
        f"Rotation Reason: {key['rotation_reason']}\n"
        f"Previous Key: {key['previous_key_id']}\n"
        f"Rotation Procedure:\n{procedure_str}"
    )
    facts = [
        {"entity": key["service"], "field": "key_id", "value": key["key_id"]},
        {"entity": key["service"], "field": "created", "value": key["created"]},
        {"entity": key["service"], "field": "expires", "value": key["expires"]},
        {"entity": key["service"], "field": "rotation_reason", "value": key["rotation_reason"]},
    ]
    return Turn(
        turn_number=turn_number,
        content=content,
        block=6,
        block_name="key_management",
        facts=facts,
    )


def _make_audit_log_turn(entries: list[dict[str, Any]], turn_number: int) -> Turn:
    """Generate a turn with a batch of audit log entries."""
    lines = []
    facts = []
    for i, entry in enumerate(entries):
        lines.append(
            f"[{entry['timestamp']}] {entry['actor']} {entry['action']} "
            f"{entry['resource']} — {entry['details']} (result: {entry['result']})"
        )
        facts.append(
            {
                "entity": f"audit_{turn_number}_{i}",
                "field": "event",
                "value": f"{entry['timestamp']}: {entry['actor']} {entry['action']} on {entry['resource']}",
            }
        )
    content = "Audit Log Entries:\n" + "\n".join(lines)
    return Turn(
        turn_number=turn_number,
        content=content,
        block=7,
        block_name="audit_logs",
        facts=facts,
    )


def _make_scan_turn(scan: dict[str, Any], turn_number: int) -> Turn:
    """Generate a turn about vulnerability scan results."""
    findings_str = "\n".join(
        f"  [{f['severity'].upper()}] {f['server']}: {f['finding']} (status: {f['status']})"
        for f in scan["findings"]
    )
    content = (
        f"Vulnerability Scan Report\n"
        f"Date: {scan['scan_date']}\n"
        f"Scanner: {scan['scanner']}\n"
        f"Scope: {scan['scope']}\n"
        f"Findings ({len(scan['findings'])} total):\n{findings_str}"
    )
    facts = []
    for f in scan["findings"]:
        facts.append(
            {
                "entity": f"scan_{scan['scan_date']}_{f['server']}",
                "field": "finding",
                "value": f"{f['finding']} (severity: {f['severity']}, status: {f['status']})",
            }
        )
    return Turn(
        turn_number=turn_number,
        content=content,
        block=8,
        block_name="scans",
        facts=facts,
    )


def _make_deployment_turn(deploy: dict[str, Any], turn_number: int) -> Turn:
    """Generate a turn about a deployment."""
    changes_str = "\n".join(f"  - {c}" for c in deploy["changes"])
    content = (
        f"Deployment Record: {deploy['server']} {deploy['version']}\n"
        f"Date: {deploy['date']}\n"
        f"Changes:\n{changes_str}"
    )
    facts = [
        {
            "entity": f"deploy_{deploy['version']}",
            "field": "server",
            "value": deploy["server"],
        },
        {
            "entity": f"deploy_{deploy['version']}",
            "field": "date",
            "value": deploy["date"],
        },
        {
            "entity": f"deploy_{deploy['version']}",
            "field": "changes",
            "value": "; ".join(deploy["changes"]),
        },
    ]
    return Turn(
        turn_number=turn_number,
        content=content,
        block=9,
        block_name="deployments",
        facts=facts,
    )


def _make_policy_turn(policy: dict[str, Any], turn_number: int) -> Turn:
    """Generate a turn about a policy document."""
    if "steps" in policy:
        body = "\n".join(f"  {s}" for s in policy["steps"])
        body += f"\n  Escalation: {policy['escalation']}"
    else:
        body = "\n".join(f"  {r}" for r in policy["rules"])
    content = (
        f"Policy Document: {policy['name']} (v{policy['version']})\n"
        f"Last Updated: {policy['last_updated']}\n"
        f"Content:\n{body}"
    )
    facts = [
        {"entity": policy["name"], "field": "version", "value": policy["version"]},
        {"entity": policy["name"], "field": "last_updated", "value": policy["last_updated"]},
    ]
    if "steps" in policy:
        for step in policy["steps"]:
            facts.append({"entity": policy["name"], "field": "step", "value": step})
    return Turn(
        turn_number=turn_number,
        content=content,
        block=10,
        block_name="policy_docs",
        facts=facts,
    )


def _make_contradiction_turn(contradiction: dict[str, Any], turn_number: int) -> Turn:
    """Generate a turn introducing contradictory information."""
    content = (
        f"Contradictory Information Detected:\n"
        f"Topic: {contradiction['topic']}\n"
        f"Source A ({contradiction['source_a']['source']}): {contradiction['source_a']['claim']}\n"
        f"Source B ({contradiction['source_b']['source']}): {contradiction['source_b']['claim']}\n"
        f"Resolution: {contradiction['resolution']}"
    )
    facts = [
        {
            "entity": f"contradiction_{turn_number}",
            "field": "topic",
            "value": contradiction["topic"],
        },
        {
            "entity": f"contradiction_{turn_number}",
            "field": "source_a",
            "value": contradiction["source_a"]["claim"],
        },
        {
            "entity": f"contradiction_{turn_number}",
            "field": "source_b",
            "value": contradiction["source_b"]["claim"],
        },
        {
            "entity": f"contradiction_{turn_number}",
            "field": "resolution",
            "value": contradiction["resolution"],
        },
    ]
    return Turn(
        turn_number=turn_number,
        content=content,
        block=11,
        block_name="contradictions",
        facts=facts,
    )


def _make_novel_api_turn(turn_number: int) -> Turn:
    """Generate a turn with novel API documentation."""
    api = NOVEL_API_DOCS
    content = (
        f"New Cloud Provider API Documentation: {api['provider']} {api['version']}\n"
        f"Monitoring Config Format: {api['monitoring_config_format']['type']}\n"
        f"Required Fields: {', '.join(api['monitoring_config_format']['required_fields'])}\n"
        f"Example Configuration:\n{api['monitoring_config_format']['example']}"
    )
    facts = [
        {"entity": "cloudwatch_pro", "field": "version", "value": api["version"]},
        {
            "entity": "cloudwatch_pro",
            "field": "config_format",
            "value": api["monitoring_config_format"]["type"],
        },
        {
            "entity": "cloudwatch_pro",
            "field": "required_fields",
            "value": ", ".join(api["monitoring_config_format"]["required_fields"]),
        },
    ]
    return Turn(
        turn_number=turn_number,
        content=content,
        block=12,
        block_name="novel_apis",
        facts=facts,
    )


def _make_network_turn(turn_number: int) -> Turn:
    """Generate a turn about network topology."""
    subnets_str = "\n".join(
        f"  {name}: {info['cidr']} — {info['purpose']}"
        for name, info in NETWORK_TOPOLOGY["subnets"].items()
    )
    lb_str = "\n".join(
        f"  {lb['name']} ({lb['type']}): {lb['ip']} → {', '.join(lb['targets'])} port {lb['port']}"
        for lb in NETWORK_TOPOLOGY["load_balancers"]
    )
    content = (
        f"Network Topology Overview\n"
        f"Subnets:\n{subnets_str}\n"
        f"Load Balancers:\n{lb_str}\n"
        f"VPN Gateway: {NETWORK_TOPOLOGY['vpn_gateway']}"
    )
    facts = []
    for name, info in NETWORK_TOPOLOGY["subnets"].items():
        facts.append({"entity": f"subnet_{name}", "field": "cidr", "value": info["cidr"]})
        facts.append({"entity": f"subnet_{name}", "field": "purpose", "value": info["purpose"]})
    for lb in NETWORK_TOPOLOGY["load_balancers"]:
        facts.append(
            {
                "entity": lb["name"],
                "field": "targets",
                "value": ", ".join(lb["targets"]),
            }
        )
    return Turn(
        turn_number=turn_number,
        content=content,
        block=1,
        block_name="infrastructure",
        facts=facts,
    )


def generate_dialogue(num_turns: int = 100, seed: int = 42) -> GroundTruth:
    """Generate deterministic security analyst dialogue.

    The dialogue covers server configs, incidents, CVEs, team info, firewall
    policies, API keys, audit logs, scans, deployments, policy docs,
    contradictions, and novel API documentation.

    Args:
        num_turns: Number of dialogue turns to generate
        seed: Random seed for reproducibility

    Returns:
        GroundTruth with turns and tracked facts
    """
    rng = random.Random(seed)
    turns: list[Turn] = []
    facts_by_entity: dict[str, list[dict[str, Any]]] = {}
    current_values: dict[str, Any] = {}

    # Build ordered content pool
    content_pool: list[Turn] = []
    turn_idx = 0

    # Infrastructure (servers + network)
    for server in SERVERS:
        content_pool.append(_make_server_turn(server, turn_idx))
        turn_idx += 1
    content_pool.append(_make_network_turn(turn_idx))
    turn_idx += 1

    # Incidents
    for incident in INCIDENTS:
        content_pool.append(_make_incident_turn(incident, turn_idx))
        turn_idx += 1

    # CVEs
    for cve in CVES:
        content_pool.append(_make_cve_turn(cve, turn_idx))
        turn_idx += 1

    # Team
    for member in TEAM:
        content_pool.append(_make_team_turn(member, turn_idx))
        turn_idx += 1

    # Firewall policies (in temporal order)
    for policy in FIREWALL_POLICIES:
        content_pool.append(_make_firewall_turn(policy, turn_idx))
        turn_idx += 1

    # API keys
    for key in API_KEYS:
        content_pool.append(_make_api_key_turn(key, turn_idx))
        turn_idx += 1

    # Audit logs (in batches of 2-3)
    for i in range(0, len(AUDIT_LOGS), 3):
        batch = AUDIT_LOGS[i : i + 3]
        content_pool.append(_make_audit_log_turn(batch, turn_idx))
        turn_idx += 1

    # Scans
    for scan in SCAN_RESULTS:
        content_pool.append(_make_scan_turn(scan, turn_idx))
        turn_idx += 1

    # Deployments
    for deploy in DEPLOYMENTS:
        content_pool.append(_make_deployment_turn(deploy, turn_idx))
        turn_idx += 1

    # Policy docs
    for policy in POLICIES:
        content_pool.append(_make_policy_turn(policy, turn_idx))
        turn_idx += 1

    # Contradictions
    for contradiction in CONTRADICTIONS:
        content_pool.append(_make_contradiction_turn(contradiction, turn_idx))
        turn_idx += 1

    # Novel API
    content_pool.append(_make_novel_api_turn(turn_idx))
    turn_idx += 1

    # Scale: repeat and shuffle to reach num_turns
    if len(content_pool) < num_turns:
        # First pass: all content in order (ensures all info is delivered)
        base_turns = list(content_pool)
        # Fill remaining with shuffled repeats
        extra_needed = num_turns - len(base_turns)
        extra_pool = list(content_pool)
        rng.shuffle(extra_pool)
        # Cycle through shuffled pool to fill
        extra_turns = []
        for i in range(extra_needed):
            extra_turns.append(extra_pool[i % len(extra_pool)])
        all_turns = base_turns + extra_turns
    else:
        all_turns = content_pool[:num_turns]

    # Renumber turns and build ground truth
    for i, turn in enumerate(all_turns):
        renumbered = Turn(
            turn_number=i,
            content=turn.content,
            block=turn.block,
            block_name=turn.block_name,
            facts=turn.facts,
        )
        turns.append(renumbered)

        for fact in turn.facts:
            entity = fact["entity"]
            if entity not in facts_by_entity:
                facts_by_entity[entity] = []
            facts_by_entity[entity].append(
                {
                    "field": fact["field"],
                    "value": fact["value"],
                    "turn": i,
                }
            )
            current_values[f"{entity}.{fact['field']}"] = fact["value"]

    return GroundTruth(
        turns=turns,
        facts_by_entity=facts_by_entity,
        current_values=current_values,
    )


# ============================================================
# Question Generation
# ============================================================


def _make_rubric(
    summary: str,
    keywords: list[str] | None = None,
    paraphrases: list[str] | None = None,
    incorrect: list[str] | None = None,
) -> GradingRubric:
    """Create a grading rubric from keywords."""
    if keywords is None:
        # Extract keywords from summary
        words = summary.replace(",", " ").replace(".", " ").split()
        keywords = [w for w in words if len(w) > 2 and not w.startswith("(")]
    return GradingRubric(
        required_keywords=keywords,
        acceptable_paraphrases=paraphrases or [],
        incorrect_patterns=incorrect or [],
    )


def _get_l1_questions(ground_truth: GroundTruth) -> list[Question]:
    """L1: Direct Recall — single-fact lookup."""
    questions = []
    delivered_entities = set()
    for turn in ground_truth.turns:
        for fact in turn.facts:
            delivered_entities.add(fact["entity"])

    # Server facts
    for server in SERVERS:
        if server["name"] in delivered_entities:
            questions.append(
                Question(
                    question_id=f"L1_port_{server['name']}",
                    text=f"What port does {server['name']} run on?",
                    expected_answer=str(server["port"]),
                    category="L1_direct_recall",
                    relevant_turns=[],
                    scoring_dimensions=["factual_accuracy"],
                    rubric=_make_rubric(str(server["port"]), keywords=[str(server["port"])]),
                )
            )
            questions.append(
                Question(
                    question_id=f"L1_ip_{server['name']}",
                    text=f"What is the IP address of {server['name']}?",
                    expected_answer=server["ip"],
                    category="L1_direct_recall",
                    relevant_turns=[],
                    scoring_dimensions=["factual_accuracy"],
                    rubric=_make_rubric(server["ip"], keywords=[server["ip"]]),
                )
            )
            questions.append(
                Question(
                    question_id=f"L1_role_{server['name']}",
                    text=f"What is the role of {server['name']}?",
                    expected_answer=server["role"],
                    category="L1_direct_recall",
                    relevant_turns=[],
                    scoring_dimensions=["factual_accuracy"],
                    rubric=_make_rubric(server["role"]),
                )
            )

    # Team facts
    for member in TEAM:
        if member["name"] in delivered_entities:
            questions.append(
                Question(
                    question_id=f"L1_oncall_{member['name'].replace(' ', '_')}",
                    text=f"What is {member['name']}'s on-call schedule?",
                    expected_answer=member["on_call"],
                    category="L1_direct_recall",
                    relevant_turns=[],
                    scoring_dimensions=["factual_accuracy"],
                    rubric=_make_rubric(member["on_call"]),
                )
            )

    # CVE facts
    for cve in CVES:
        if cve["id"] in delivered_entities:
            questions.append(
                Question(
                    question_id=f"L1_cvss_{cve['id']}",
                    text=f"What is the CVSS score of {cve['id']}?",
                    expected_answer=str(cve["cvss"]),
                    category="L1_direct_recall",
                    relevant_turns=[],
                    scoring_dimensions=["factual_accuracy"],
                    rubric=_make_rubric(str(cve["cvss"]), keywords=[str(cve["cvss"])]),
                )
            )

    return questions


def _get_l2_questions(ground_truth: GroundTruth) -> list[Question]:
    """L2: Multi-Source Synthesis — combine info from multiple turns."""
    questions = []

    for cve in CVES:
        servers = cve["affected_servers"]
        questions.append(
            Question(
                question_id=f"L2_affected_{cve['id']}",
                text=f"Which servers are affected by {cve['id']}?",
                expected_answer=", ".join(servers),
                category="L2_multi_source_synthesis",
                relevant_turns=[],
                scoring_dimensions=["factual_accuracy", "specificity"],
                rubric=_make_rubric(", ".join(servers), keywords=servers),
            )
        )

    # Cross-reference: which team member handles which incident?
    questions.append(
        Question(
            question_id="L2_incident_responders",
            text="Who responded to INC-2024-001 and what was their role?",
            expected_answer="Bob Chen (SOC Analyst) acknowledged the initial alert; the server was isolated and rebuilt. EDR caught the Lockbit payload.",
            category="L2_multi_source_synthesis",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "Bob Chen, SOC Analyst, EDR, Lockbit",
                keywords=["Bob Chen", "EDR"],
            ),
        )
    )

    # Combine scan findings with CVE data
    questions.append(
        Question(
            question_id="L2_unpatched_servers",
            text="Which servers have unpatched high-severity vulnerabilities according to the latest scan?",
            expected_answer="staging-db-01 (CVE-2024-9012, PostgreSQL) and staging-app-01 (CVE-2024-5678, kernel)",
            category="L2_multi_source_synthesis",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "staging-db-01, staging-app-01",
                keywords=["staging-db-01", "staging-app-01"],
            ),
        )
    )

    return questions


def _get_l3_questions(ground_truth: GroundTruth) -> list[Question]:
    """L3: Temporal Reasoning — track changes over time."""
    questions = [
        Question(
            question_id="L3_firewall_evolution",
            text="How has the firewall policy changed over the last 3 updates?",
            expected_answer=(
                "v1 (2024-01-01): Baseline with standard allow/deny rules. "
                "v2 (2024-03-10): Added rate limiting (100 req/s), UDP flood protection, and geo-blocking after DDoS. "
                "v3 (2024-03-20): Blocked C2 IPs, added egress filtering, TLS inspection, and restricted staging outbound after exfiltration."
            ),
            category="L3_temporal_reasoning",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "temporal_awareness"],
            rubric=_make_rubric(
                "v1 baseline, v2 rate limiting DDoS, v3 egress C2",
                keywords=["v1", "v2", "v3", "rate limiting", "egress"],
            ),
        ),
        Question(
            question_id="L3_incident_chronology",
            text="List all security incidents in chronological order with their detection dates.",
            expected_answer=(
                "INC-2024-001 (2024-01-15): Ransomware attempt. "
                "INC-2024-002 (2024-02-03): API key exfiltration. "
                "INC-2024-003 (2024-03-10): DDoS attack."
            ),
            category="L3_temporal_reasoning",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "temporal_awareness"],
            rubric=_make_rubric(
                "INC-2024-001 Jan, INC-2024-002 Feb, INC-2024-003 Mar",
                keywords=["INC-2024-001", "INC-2024-002", "INC-2024-003"],
            ),
        ),
        Question(
            question_id="L3_app_deploy_history",
            text="What versions has prod-app-01 been deployed at and when?",
            expected_answer="v2.3 on 2024-01-05 (OAuth, session fix), then v2.4 on 2024-03-12 (rate limiting, SameSite fix, tracing, Jackson patch)",
            category="L3_temporal_reasoning",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "temporal_awareness"],
            rubric=_make_rubric(
                "v2.3 January, v2.4 March",
                keywords=["v2.3", "v2.4"],
            ),
        ),
    ]
    return questions


def _get_l4_questions(ground_truth: GroundTruth) -> list[Question]:
    """L4: Procedural Knowledge — recall multi-step procedures."""
    questions = [
        Question(
            question_id="L4_rotate_payment_key",
            text="What are the steps to rotate the API key for the service-payment service?",
            expected_answer=(
                "1. Generate new key in payment provider dashboard. "
                "2. Update key in HashiCorp Vault at secret/prod/payment-api-key. "
                "3. Trigger rolling restart of prod-app-01. "
                "4. Verify payment processing with test transaction. "
                "5. Revoke old key in payment provider dashboard. "
                "6. Update rotation log in Confluence."
            ),
            category="L4_procedural",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "Vault, rolling restart, test transaction, revoke old key",
                keywords=["Vault", "restart", "revoke"],
            ),
        ),
        Question(
            question_id="L4_incident_response_steps",
            text="What are the 7 steps of the incident response procedure?",
            expected_answer=(
                "1. DETECT (acknowledge within 15 min). "
                "2. TRIAGE (classify severity within 30 min). "
                "3. CONTAIN (isolate affected systems). "
                "4. INVESTIGATE (capture forensic evidence). "
                "5. ERADICATE (remove threat access, patch). "
                "6. RECOVER (restore from clean backups). "
                "7. LESSONS LEARNED (post-incident review within 5 days)."
            ),
            category="L4_procedural",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "DETECT TRIAGE CONTAIN INVESTIGATE ERADICATE RECOVER LESSONS",
                keywords=["DETECT", "TRIAGE", "CONTAIN", "INVESTIGATE", "ERADICATE", "RECOVER"],
            ),
        ),
    ]
    return questions


def _get_l5_questions(ground_truth: GroundTruth) -> list[Question]:
    """L5: Contradiction Resolution — resolve conflicting information."""
    questions = [
        Question(
            question_id="L5_patch_contradiction",
            text=(
                "The INC-2024-001 post-mortem says prod-app-01 was rebuilt with latest patches "
                "including the CVE-2024-1234 fix, but the Nessus scan shows the Java runtime "
                "still has CVE-2024-1234 exposure. Which is correct?"
            ),
            expected_answer=(
                "Both are partially correct. The OS-level OpenSSL was patched via the golden image rebuild, "
                "but the Java runtime bundles its own OpenSSL which remains at the old version. The risk is "
                "accepted because TLS terminates at Nginx, not at the Java application layer."
            ),
            category="L5_contradiction",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "Both partially correct, OS patched, Java bundles own OpenSSL, TLS at Nginx",
                keywords=["OS", "Java", "Nginx", "accepted"],
            ),
        ),
        Question(
            question_id="L5_staging_security",
            text=(
                "The patch management policy says high-severity CVEs must be patched within 7 days. "
                "The scan shows staging-db-01 has been unpatched for CVE-2024-9012 (published March 1, "
                "scan March 15 — 14 days). Is the environment compliant?"
            ),
            expected_answer=(
                "No, staging-db-01 is non-compliant. CVE-2024-9012 is high severity (CVSS 8.4) "
                "and was published on March 1. The scan on March 15 (14 days later) shows it is still "
                "unpatched, exceeding the 7-day policy. staging-app-01 is also non-compliant for CVE-2024-5678."
            ),
            category="L5_contradiction",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "non-compliant, 14 days, 7-day policy, staging-db-01",
                keywords=["non-compliant", "14 days", "7"],
            ),
        ),
    ]
    return questions


def _get_l6_questions(ground_truth: GroundTruth) -> list[Question]:
    """L6: Incremental Update — track what changed between versions."""
    questions = [
        Question(
            question_id="L6_deploy_changes",
            text="The deployment was updated from v2.3 to v2.4 — what changed?",
            expected_answer=(
                "v2.4 added: rate limiting middleware (100 req/s per IP using Redis), "
                "fixed session cookie SameSite (Lax → Strict), "
                "added request ID header for distributed tracing, "
                "upgraded Jackson JSON library to 2.16.1 (security patch)."
            ),
            category="L6_incremental_update",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "rate limiting, SameSite Strict, request ID, Jackson 2.16.1",
                keywords=["rate limiting", "SameSite", "Strict", "Jackson"],
            ),
        ),
        Question(
            question_id="L6_firewall_v2_changes",
            text="What specific changes were made in firewall policy v2 compared to v1?",
            expected_answer=(
                "v2 added: rate limiting (100 req/s per source IP on port 443), "
                "UDP flood protection (drop UDP > 1000/s from single source), "
                "and geo-blocking for known botnet source countries."
            ),
            category="L6_incremental_update",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "100 req/s, UDP 1000/s, geo-blocking",
                keywords=["100", "UDP", "geo-blocking"],
            ),
        ),
    ]
    return questions


def _get_l7_questions(ground_truth: GroundTruth) -> list[Question]:
    """L7: Teaching / Explanation — explain concepts to someone new."""
    questions = [
        Question(
            question_id="L7_incident_response_teach",
            text="Explain the incident response procedure to a new team member joining the security team.",
            expected_answer=(
                "Our incident response has 7 steps: First DETECT — you acknowledge alerts within 15 minutes. "
                "Then TRIAGE — classify how severe it is within 30 minutes. CONTAIN — isolate affected systems "
                "to stop the spread. INVESTIGATE — capture forensic evidence and find the root cause. "
                "ERADICATE — remove the attacker's access and patch the vulnerability. RECOVER — restore "
                "from clean backups. Finally, LESSONS LEARNED — we do a post-incident review within 5 business days. "
                "For critical incidents, page the Security Lead immediately; high severity notify within 1 hour."
            ),
            category="L7_teaching",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity", "source_attribution"],
            rubric=_make_rubric(
                "7 steps, DETECT, TRIAGE, CONTAIN, INVESTIGATE, ERADICATE, RECOVER, LESSONS",
                keywords=["DETECT", "TRIAGE", "CONTAIN", "RECOVER"],
            ),
        ),
        Question(
            question_id="L7_network_layout_teach",
            text="Describe our network layout to a new team member — what subnets do we have and what goes where?",
            expected_answer=(
                "We have 4 subnets: DMZ (10.0.0.0/24) for internet-facing services and the bastion host. "
                "Production (10.0.1.0/24) for live workloads — database, web, app, and cache servers. "
                "Staging (10.0.2.0/24) for pre-production testing. Management (10.0.3.0/24) for monitoring "
                "and admin tools. Traffic from the internet goes through load balancers in DMZ to production servers."
            ),
            category="L7_teaching",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "4 subnets, DMZ 10.0.0.0, production 10.0.1.0, staging 10.0.2.0, management 10.0.3.0",
                keywords=["DMZ", "10.0.0.0", "10.0.1.0", "10.0.2.0", "10.0.3.0"],
            ),
        ),
    ]
    return questions


def _get_l8_questions(ground_truth: GroundTruth) -> list[Question]:
    """L8: Confidence Calibration — assess certainty about security posture."""
    questions = [
        Question(
            question_id="L8_perimeter_confidence",
            text="How confident should we be that the perimeter is secure?",
            expected_answer=(
                "Moderate confidence with known gaps. Production servers are well-patched and firewalled "
                "(v3 with egress filtering, C2 blocking, rate limiting). However: bastion-01 still allows "
                "SSH password auth (should be key-only), staging servers have unpatched high-severity CVEs "
                "(CVE-2024-5678, CVE-2024-9012), and TLS 1.0 is still enabled on prod-web-01. "
                "The perimeter is significantly hardened after three incidents but has measurable gaps."
            ),
            category="L8_confidence",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "confidence_calibration"],
            rubric=_make_rubric(
                "moderate, bastion password auth, staging unpatched, TLS 1.0",
                keywords=["bastion", "staging", "TLS 1.0", "unpatched"],
            ),
        ),
        Question(
            question_id="L8_data_exfil_confidence",
            text=(
                "After INC-2024-002, how confident can we be that no further data was exfiltrated "
                "beyond the 47 detected API calls?"
            ),
            expected_answer=(
                "Limited confidence. We confirmed 47 unauthorized API calls from IP 203.0.113.50 using "
                "the compromised service-payment key. The key was revoked, the IP blocked, and audit logs "
                "reviewed. However, we cannot be fully confident because: the key was exposed via a public "
                "GitHub Actions log (unknown exposure window), other IPs may have used the key before detection, "
                "and debug logging may have exposed other secrets. Secret scanning was enabled post-incident."
            ),
            category="L8_confidence",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "confidence_calibration"],
            rubric=_make_rubric(
                "47 calls, 203.0.113.50, limited confidence, unknown exposure window",
                keywords=["47", "203.0.113.50", "limited"],
            ),
        ),
    ]
    return questions


def _get_l9_questions(ground_truth: GroundTruth) -> list[Question]:
    """L9: Causal Reasoning — identify root causes."""
    questions = [
        Question(
            question_id="L9_outage_cause",
            text="What caused the outage on 2024-01-15?",
            expected_answer=(
                "An employee clicked a phishing link that delivered a Lockbit ransomware payload to prod-app-01. "
                "The spear-phishing email bypassed the email gateway. EDR detected and blocked the payload "
                "before encryption started, but the server had to be isolated and rebuilt, causing 2 hours of "
                "application downtime. Root cause: email gateway filtering gap + employee susceptibility to phishing."
            ),
            category="L9_causal",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "phishing, Lockbit, EDR blocked, email gateway, 2 hours downtime",
                keywords=["phishing", "Lockbit", "EDR", "email gateway"],
            ),
        ),
        Question(
            question_id="L9_key_compromise_cause",
            text="What was the chain of events that led to the API key compromise in INC-2024-002?",
            expected_answer=(
                "A developer committed a .env file with a production API key (service-payment) to a public "
                "GitHub fork. The CI/CD pipeline (GitHub Actions) had debug logging enabled, which printed the "
                "key in the build log. An external actor (IP 203.0.113.50) found the exposed key and made 47 "
                "unauthorized API calls before detection at 11:15 UTC."
            ),
            category="L9_causal",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                ".env file, public fork, debug logging, 203.0.113.50, 47 calls",
                keywords=[".env", "public", "debug", "203.0.113.50"],
            ),
        ),
    ]
    return questions


def _get_l10_questions(ground_truth: GroundTruth) -> list[Question]:
    """L10: Counterfactual Reasoning — what-if analysis."""
    questions = [
        Question(
            question_id="L10_no_patch_cve5678",
            text="What would have happened if we hadn't patched CVE-2024-5678 on production servers?",
            expected_answer=(
                "CVE-2024-5678 is a Linux kernel privilege escalation (CVSS 7.8) allowing local root access "
                "via crafted netfilter rules. Without patching, any attacker who gained user-level access "
                "(e.g., via a future phishing incident like INC-2024-001) could escalate to root on all "
                "production servers. This would mean full control over prod-db-01 (database with customer data), "
                "prod-web-01, prod-app-01, and prod-cache-01. The ransomware in INC-2024-001 was caught by EDR "
                "but if the attacker had root access via this CVE, they could have disabled EDR first."
            ),
            category="L10_counterfactual",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "privilege escalation, root access, disable EDR, all production servers",
                keywords=["privilege escalation", "root", "EDR"],
            ),
        ),
        Question(
            question_id="L10_no_rate_limit",
            text="What would have happened during the DDoS if we hadn't deployed rate limiting?",
            expected_answer=(
                "Without rate limiting in firewall v2, the DDoS attack (2.5 Gbps UDP flood) would have continued "
                "to saturate prod-web-01. The 503 errors lasted 45 minutes with mitigation; without it, the outage "
                "could have been much longer until the ISP upstream filtering took effect. All web traffic "
                "would have been affected since prod-web-01 is the sole TLS termination point behind alb-prod-web."
            ),
            category="L10_counterfactual",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "2.5 Gbps, UDP flood, longer outage, prod-web-01, sole TLS termination",
                keywords=["2.5 Gbps", "UDP", "outage", "TLS termination"],
            ),
        ),
    ]
    return questions


def _get_l11_questions(ground_truth: GroundTruth) -> list[Question]:
    """L11: Novel Skill Application — apply learned patterns to new APIs."""
    questions = [
        Question(
            question_id="L11_monitoring_config",
            text=(
                "Given the CloudWatch Pro API v3 documentation we received, write a monitoring "
                "configuration that alerts when prod-db-01 CPU exceeds 80% for 5 minutes "
                "and when disk usage exceeds 90% for 10 minutes."
            ),
            expected_answer=(
                "monitors:\n"
                "  - name: \"prod-db-01-cpu\"\n"
                "    metric: \"system.cpu.percent\"\n"
                "    threshold: 80\n"
                "    duration: \"5m\"\n"
                "    severity: \"warning\"\n"
                "    notify: [\"slack:#ops-alerts\"]\n"
                "  - name: \"prod-db-01-disk\"\n"
                "    metric: \"system.disk.used_percent\"\n"
                "    threshold: 90\n"
                "    duration: \"10m\"\n"
                "    severity: \"critical\"\n"
                "    notify: [\"pagerduty:security-oncall\"]"
            ),
            category="L11_novel_skill",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "monitors, cpu.percent, 80, 5m, disk.used_percent, 90, 10m",
                keywords=["cpu", "80", "5m", "disk", "90", "10m"],
            ),
        ),
    ]
    return questions


def _get_l12_questions(ground_truth: GroundTruth) -> list[Question]:
    """L12: Far Transfer — apply patterns from one domain to another."""
    questions = [
        Question(
            question_id="L12_transfer_incident_to_network",
            text=(
                "Apply the debugging pattern we used for the database-related incident (INC-2024-002 "
                "API key compromise via CI/CD) to diagnose a hypothetical network issue where "
                "prod-web-01 starts sending traffic to unknown external IPs."
            ),
            expected_answer=(
                "Following the same debugging pattern from INC-2024-002: "
                "1. DETECT — Check monitoring alerts for anomalous outbound traffic (like we caught the 47 API calls). "
                "2. Identify the source — determine which process on prod-web-01 is making connections (like we traced to CI debug logs). "
                "3. Block immediately — add the unknown IPs to the firewall block list (like we blocked 203.0.113.50 at WAF). "
                "4. Audit trail — check audit logs for recent changes (deploys, config changes) that could have introduced the behavior. "
                "5. Root cause — look for compromised credentials or malicious code (like the .env exposure). "
                "6. Remediate — patch the vector, rotate credentials, update egress filtering (like firewall v3 changes). "
                "7. Post-incident — review within 5 days per our incident response procedure."
            ),
            category="L12_far_transfer",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "detect, block, audit, root cause, remediate, INC-2024-002 pattern",
                keywords=["audit", "block", "root cause", "remediate"],
            ),
        ),
        Question(
            question_id="L12_transfer_patching_to_keys",
            text=(
                "Our patch management policy defines SLA timelines based on severity. "
                "Apply the same tiered-urgency approach to API key rotation — "
                "how should we prioritize key rotations based on exposure level?"
            ),
            expected_answer=(
                "Adapting the patch management SLA tiers to key rotation: "
                "Critical (key confirmed compromised, like service-payment in INC-2024-002): rotate within 1 hour, "
                "revoke immediately. "
                "High (key potentially exposed, e.g., found in logs or shared repo): rotate within 24 hours. "
                "Medium (key approaching expiration or used by deprecated service): rotate within 7 days. "
                "Low (scheduled rotation per 180-day policy): rotate in next maintenance window. "
                "This mirrors the patch policy's CVSS-based tiers: critical 48h, high 7d, medium 30d, low next window."
            ),
            category="L12_far_transfer",
            relevant_turns=[],
            scoring_dimensions=["factual_accuracy", "specificity"],
            rubric=_make_rubric(
                "tiered urgency, critical 1 hour, high 24 hours, mirrors patch policy",
                keywords=["tiered", "critical", "1 hour", "24 hours"],
            ),
        ),
    ]
    return questions


# Level generator dispatch
_LEVEL_GENERATORS = {
    "L1": _get_l1_questions,
    "L2": _get_l2_questions,
    "L3": _get_l3_questions,
    "L4": _get_l4_questions,
    "L5": _get_l5_questions,
    "L6": _get_l6_questions,
    "L7": _get_l7_questions,
    "L8": _get_l8_questions,
    "L9": _get_l9_questions,
    "L10": _get_l10_questions,
    "L11": _get_l11_questions,
    "L12": _get_l12_questions,
}


def generate_questions(
    ground_truth: GroundTruth, num_questions: int = 50
) -> list[Question]:
    """Generate L1-L12 questions from the security analyst scenario.

    Distributes questions across levels according to LEVEL_DISTRIBUTION,
    ensuring at least 1 question per level when num_questions >= 12.

    Args:
        ground_truth: Ground truth from generate_dialogue()
        num_questions: Target number of questions

    Returns:
        List of Question objects tagged with L1-L12 categories
    """
    all_questions: list[Question] = []

    for level, generator_fn in _LEVEL_GENERATORS.items():
        target_count = max(1, int(num_questions * LEVEL_DISTRIBUTION[level]))
        candidates = generator_fn(ground_truth)
        all_questions.extend(candidates[:target_count])

    # Trim or pad to exact count
    if len(all_questions) > num_questions:
        all_questions = all_questions[:num_questions]
    elif len(all_questions) < num_questions:
        # Fill from L1 (most questions available)
        extra = _get_l1_questions(ground_truth)
        for q in extra:
            if q.question_id not in {eq.question_id for eq in all_questions}:
                all_questions.append(q)
                if len(all_questions) >= num_questions:
                    break

    # Ensure all questions have rubrics
    for q in all_questions:
        if q.rubric is None:
            q.rubric = _make_rubric(q.expected_answer)

    return all_questions[:num_questions]


__all__ = [
    "LEVEL_NAMES",
    "LEVEL_DISTRIBUTION",
    "SERVERS",
    "NETWORK_TOPOLOGY",
    "INCIDENTS",
    "CVES",
    "SCAN_RESULTS",
    "TEAM",
    "FIREWALL_POLICIES",
    "API_KEYS",
    "AUDIT_LOGS",
    "DEPLOYMENTS",
    "POLICIES",
    "CONTRADICTIONS",
    "NOVEL_API_DOCS",
    "generate_dialogue",
    "generate_questions",
]
