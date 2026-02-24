# Evaluation Categories

This document provides a detailed explanation of every evaluation category in `amplihack-agent-eval`, from the progressive levels (L1--L16) to the long-horizon question categories.

## Progressive Levels (L1--L12)

These are hand-crafted evaluation levels with curated articles and questions. Each level isolates a specific cognitive capability.

### L1: Single Source Direct Recall

**Difficulty**: 1/5 | **Category**: memory | **Prerequisites**: none

**What it tests**: The most basic memory operation -- retrieving a fact that was directly stated in a single source. No inference, no synthesis, no temporal reasoning required.

**Example articles**: Medal standings from a sporting event with specific numbers.

**Example questions**:
- "How many total medals does Norway have as of February 15?" (Expected: "26 total medals")
- "Which country is in second place?" (Expected: "Italy with 22 total medals")

**Reasoning types**: `direct_recall`

**Why it matters**: This is the baseline. If an agent fails L1, its memory system has fundamental issues with storage or retrieval. Every higher level depends on this capability.

### L2: Multi-Source Synthesis

**Difficulty**: 2/5 | **Category**: memory | **Prerequisites**: L1

**What it tests**: Combining information from 2--3 independent sources to answer a question that no single source can answer alone.

**Example articles**: Medal standings + individual athlete achievements + historical context (3 separate articles).

**Example questions**:
- "How does Italy's 2026 gold medal performance compare to their previous best?" (requires standings article + historical article)
- "Which country's individual athletes won the most medals mentioned in the athlete achievements article?" (requires counting across sources)

**Reasoning types**: `cross_source_synthesis`

**Why it matters**: Real-world information rarely comes from a single source. Agents must integrate across sources to provide complete answers.

### L3: Temporal Reasoning

**Difficulty**: 3/5 | **Category**: memory | **Prerequisites**: L1

**What it tests**: Tracking values that change over time and computing differences between temporal snapshots.

**Example articles**: Medal standings on Day 7, Day 9, and Day 10 (three timestamped updates).

**Example questions**:
- "How many medals did Norway win between Day 7 and Day 9?" (Expected: "8 medals (from 18 to 26)")
- "Describe the trend in Italy's gold medal performance over the three days" (Expected: "+3 golds Day 7-9, then +1 gold Day 9-10, slowing")

**Reasoning types**: `temporal_difference`, `temporal_comparison`, `temporal_trend`

**Requires**: `temporal_ordering = True`

**Why it matters**: Agents operating in dynamic environments must understand not just current state but how things have changed.

### L4: Procedural Learning

**Difficulty**: 2/5 | **Category**: memory | **Prerequisites**: L1

**What it tests**: Learning a multi-step procedure from documentation and reproducing or applying it.

**Example articles**: A detailed Flutter development setup guide with 9 numbered steps and troubleshooting tips.

**Example questions**:
- "What command creates a new Flutter project?" (Expected: "flutter create my_app")
- "Describe the complete workflow from creating a project to running tests" (requires recalling the ordered steps)
- "If I want to create a project called 'weather_app' and add the http package, what exact commands would I run?" (requires applying the procedure to a new scenario)

**Reasoning types**: `procedural_recall`, `procedural_troubleshooting`, `procedural_sequence`, `procedural_application`

**Why it matters**: Many real tasks involve following procedures. Agents must be able to learn, recall, and adapt procedures to new contexts.

### L5: Contradiction Handling

**Difficulty**: 3/5 | **Category**: memory | **Prerequisites**: L1, L2

**What it tests**: Detecting when two or more sources provide conflicting information and reasoning about the conflict.

**Example articles**: Two articles about Olympics viewership -- one from the IOC claiming 1.2 billion viewers, another from independent analysts claiming 800 million.

**Example questions**:
- "How many people watched the 2026 opening ceremony?" (Expected: "Conflicting reports: IOC estimates 1.2 billion, independent analysts report 800 million")
- "Which viewership figure would you consider more reliable and why?" (requires source credibility reasoning)

**Reasoning types**: `contradiction_detection`, `contradiction_reasoning`, `source_credibility`

**Why it matters**: In real-world information environments, sources frequently disagree. An agent that picks one value without acknowledging the conflict is unreliable. An agent that acknowledges both values and reasons about the discrepancy demonstrates understanding.

### L6: Incremental Learning

**Difficulty**: 3/5 | **Category**: memory | **Prerequisites**: L1

**What it tests**: Updating knowledge when new information supersedes old information.

**Example articles**: An article from Feb 15 stating an athlete has 9 gold medals, followed by a Feb 17 article updating the count to 10.

**Example questions**:
- "How many Olympic gold medals does Johannes Klaebo have?" (Expected: "10" -- must use the update, not the original)
- "How did Klaebo's record change between February 15 and February 17?" (Expected: "Increased from 9 to 10")

**Reasoning types**: `incremental_update`, `incremental_tracking`, `incremental_synthesis`

**Requires**: `update_handling = True`

**Why it matters**: An agent that returns stale data when a newer value exists is dangerous, especially in time-sensitive domains. This level tests the critical ability to supersede outdated information.

### L7: Teacher-Student Knowledge Transfer

**Difficulty**: 3/5 | **Category**: memory | **Prerequisites**: L1, L2

**What it tests**: The agent learns material and then must "teach" it accurately. This tests depth of understanding -- superficial memorization is not sufficient for accurate teaching.

**Example**: The agent learns multi-source Olympic data, then answers questions as if explaining to someone who has not read the articles.

**Reasoning types**: `knowledge_transfer_recall`, `knowledge_transfer_synthesis`

**Why it matters**: Based on the educational principle that the best test of knowledge is being able to teach it (Chi, 1994). An agent that can teach accurately has genuinely internalized the material.

### L8: Metacognition / Confidence Calibration

**Difficulty**: 4/5 | **Category**: reasoning | **Prerequisites**: L1

**What it tests**: Whether the agent knows what it knows and what it does not know. Can it calibrate its confidence appropriately?

**Example questions**:
- "How confident should you be in answering 'How many medals does Canada have?'" (Expected: "Low confidence -- Canada is not in the data")
- "Which of these questions can you answer with HIGH confidence and which with LOW?" (requires distinguishing answerable from unanswerable)

**Reasoning types**: `confidence_calibration`, `gap_identification`, `confidence_discrimination`

**Why it matters**: Based on the MUSE framework (arXiv 2024) and Dunlosky & Metcalfe (2009). Overconfident agents are dangerous -- they present guesses as facts. Well-calibrated agents are trustworthy because they say "I don't know" when appropriate.

### L9: Causal Reasoning

**Difficulty**: 4/5 | **Category**: reasoning | **Prerequisites**: L1, L3

**What it tests**: Identifying causal chains and root causes from correlated observations.

**Example**: An article describing Italy's Olympic improvement, listing 5 contributing factors with causal relationships between them.

**Example questions**:
- "What caused Italy to improve from 3 golds in 2018 to 8 golds in 2026?" (requires tracing the causal chain)
- "Which single factor was most important for Italy's improvement?" (requires identifying the root cause)

**Reasoning types**: `causal_chain`, `counterfactual_causal`, `root_cause_analysis`

**Why it matters**: Based on Pearl's causal hierarchy (2009). Correlation is not causation -- agents must be able to trace cause-and-effect relationships to provide useful analysis.

### L10: Counterfactual Reasoning

**Difficulty**: 5/5 | **Category**: reasoning | **Prerequisites**: L1, L9

**What it tests**: "What if X didn't happen?" reasoning about hypothetical alternatives.

**Example questions**:
- "If Klaebo had not competed, would Norway still have led the gold medal count?" (requires computing the counterfactual medal standings)
- "In a world where cross-country skiing was removed from the Olympics, how would standings change?" (requires structural counterfactual)

**Reasoning types**: `counterfactual_removal`, `counterfactual_timing`, `counterfactual_structural`

**Why it matters**: Based on Byrne (2005) and the nearest-possible-world constraint. Counterfactual reasoning is essential for planning, risk assessment, and decision making.

### L11: Novel Skill Acquisition

**Difficulty**: 5/5 | **Category**: reasoning | **Prerequisites**: L1, L4

**What it tests**: Learning a genuinely new skill from documentation and applying it to solve problems. The skill must be post-training-cutoff to ensure the agent is actually learning, not recalling.

**Example**: GitHub Agentic Workflows (gh-aw) documentation, including frontmatter syntax, compilation rules, and security architecture.

**Example questions**:
- "Write a gh-aw workflow file that runs on PR creation and checks for security issues" (requires applying learned syntax)
- "What happens if you edit the markdown body of a workflow -- do you need to recompile?" (requires understanding compilation rules)

**Reasoning types**: `skill_application`, `novel_syntax`, `skill_troubleshooting`

**Why it matters**: The true test of an agent's learning capability is not recall of training data but the ability to learn genuinely new skills from documentation encountered at runtime.

### L12: Far Transfer

**Difficulty**: 5/5 | **Category**: reasoning | **Prerequisites**: L1, L9, L11

**What it tests**: Applying reasoning patterns learned in one domain to solve problems in a completely different domain.

**Example**: Learn Olympic medal analysis patterns, then apply the same analytical framework to a business scenario.

**Reasoning types**: `cross_domain_transfer`, `analogical_reasoning`

**Why it matters**: Transfer learning is the hallmark of genuine understanding. An agent that can abstract patterns and apply them to new domains demonstrates deep reasoning rather than pattern matching.

## Extended Levels (L13--L16)

These levels extend the evaluation beyond memory and reasoning into tool use, information management, robustness, and decision making.

### L13: Tool Selection

**Difficulty**: 3/5 | **Category**: tool_use

**What it tests**: Given a set of available tools, can the agent select the right ones and chain them in the correct order?

**Metrics**:
- `tool_selection_accuracy` -- Did the agent pick the right tools? (Jaccard similarity)
- `tool_efficiency` -- Were there unnecessary tool calls? (optimal / actual ratio)
- `tool_chain_correctness` -- Was the ordering correct? (longest common subsequence)

**Scenarios**: Multi-domain scenarios (DevOps, data analysis, security) with expected tool sequences.

**Example**: "Deploy a new version of the API service" -- expected tools: `check_tests`, `build_container`, `push_registry`, `update_deployment`, `verify_health`

**Why it matters**: Real agents must select and orchestrate tools. Tool selection errors compound -- picking the wrong tool at step 1 invalidates the entire chain.

### L14: Selective Forgetting

**Difficulty**: 3/5 | **Category**: memory

**What it tests**: When facts are updated, does the agent return the current value and not present the old value as current?

**Metrics**:
- `current_value_accuracy` -- Does the agent return the current value?
- `stale_data_penalty` -- Does it present old values as current?
- `update_awareness` -- Does it acknowledge that the value was updated?

**Example**: Server configuration changes from 16GB RAM to 32GB RAM. The agent should return 32GB and ideally note the upgrade.

**Why it matters**: An agent that remembers everything indiscriminately is dangerous. The ability to deprioritize superseded information is as important as recall. This is distinct from L6 (incremental learning) -- L14 specifically penalizes presenting stale data as current.

### L15: Adversarial Recall

**Difficulty**: 4/5 | **Category**: reasoning

**What it tests**: Resistance to hallucination when asked plausible-but-wrong questions about information not in the knowledge base.

**Metrics**:
- `hallucination_resistance` -- Does the agent say "I don't know" when appropriate?
- `fact_boundary_awareness` -- Can it distinguish known from unknown?
- `confidence_calibration` -- Is it appropriately uncertain?

**Scenarios**: Questions that reference entities similar to but different from what the agent has learned (e.g., asking about "Project Falcon" when the agent only knows about "Project Atlas").

**Why it matters**: Agents that always provide an answer, even when they do not have the information, are unreliable and potentially harmful. Honest uncertainty is a sign of intelligence, not weakness.

### L16: Decision From Memory

**Difficulty**: 5/5 | **Category**: reasoning

**What it tests**: The highest cognitive level -- can the agent recall facts, analyze them, and make correct decisions?

**Metrics**:
- `decision_quality` -- Is the decision correct given the available facts?
- `reasoning_quality` -- Does the explanation reference the correct facts?
- `fact_usage` -- Were the right facts used to support the decision?

**Scenarios**: Decision problems across domains (hiring, infrastructure planning, security response) where the correct decision depends on facts the agent has previously learned.

**Example**: "Based on the project timelines and team availability you've learned about, which project should receive additional resources?" (requires recalling multiple project states and reasoning about the best allocation)

**Why it matters**: Memory without application is useless. The ultimate test is whether stored knowledge leads to good decisions.

## Long-Horizon Question Categories

These categories are generated dynamically from the 12-block dialogue in the long-horizon evaluation. They overlap conceptually with the progressive levels but are tested at much larger scale.

### `needle_in_haystack`
Direct fact retrieval from a specific person, project, or technical domain -- finding one fact among potentially thousands of turns. Tests basic retrieval fidelity.

### `temporal_evolution`
Tracking changes to projects (deadlines, budgets, leads) and the evolving story. Questions require knowing not just the current value but the history of changes.

### `numerical_precision`
Exact recall of specific numbers, percentages, monetary values, and metrics. No approximation accepted.

### `source_attribution`
Identifying which source made a specific claim, especially when multiple sources discuss the same topic with different values.

### `cross_reference`
Connecting facts across blocks -- e.g., linking a person from Block 1 to a project from Block 2 to a technical fact from Block 3.

### `distractor_resistance`
Answering questions accurately when irrelevant fun facts from Block 8 might confuse retrieval.

### `meta_memory`
Questions about what the agent knows -- counts, categories, completeness of knowledge.

### `security_log_analysis`
Pattern recognition in structured security events -- identifying attack sequences, mapping IP addresses to events, recognizing known attack patterns.

### `incident_tracking`
Following incident timelines from creation through investigation to resolution.

### `infrastructure_knowledge`
Recall of specific infrastructure configuration details.

### `problem_solving`
Applying stored problem-solution knowledge to questions.

### `multi_hop_reasoning`
Questions requiring 2+ retrieval steps to compose the answer from multiple facts.

## Category-to-Skill Mapping

| Category                | Primary Skill          | Secondary Skill           |
|-------------------------|------------------------|---------------------------|
| `needle_in_haystack`    | Retrieval precision    | Entity recognition        |
| `temporal_evolution`    | Temporal ordering      | Change tracking           |
| `numerical_precision`   | Exact recall           | Number handling           |
| `source_attribution`    | Provenance tracking    | Source disambiguation     |
| `cross_reference`       | Graph traversal        | Entity linking            |
| `distractor_resistance` | Relevance filtering    | Confidence weighting      |
| `meta_memory`           | Self-awareness         | Aggregation               |
| `security_log_analysis` | Pattern recognition    | Structured data parsing   |
| `incident_tracking`     | State machine tracking | Timeline reconstruction   |
| `infrastructure_knowledge` | Configuration recall | Specification matching    |
| `problem_solving`       | Solution retrieval     | Analogical reasoning      |
| `multi_hop_reasoning`   | Compositional recall   | Fact chaining             |
| L13 Tool Selection      | Planning               | Tool knowledge            |
| L14 Selective Forgetting| Update management      | Stale data detection      |
| L15 Adversarial Recall  | Honesty                | Boundary awareness        |
| L16 Decision From Memory| Application            | Analytical reasoning      |
