# amplihack-agent-eval Documentation

Comprehensive documentation for the `amplihack-agent-eval` evaluation framework for goal-seeking AI agents.

## Comprehensive Guides

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Package structure, core components, design principles, data flow diagrams |
| [LONG_HORIZON_EVAL.md](LONG_HORIZON_EVAL.md) | Long-horizon memory evaluation: 12 blocks, 15 question categories, grading system |
| [CATEGORIES.md](CATEGORIES.md) | All evaluation categories: L1--L16 progressive levels and long-horizon categories |
| [EXTENDING.md](EXTENDING.md) | How to extend the framework: custom adapters, new levels, data generators, grading |
| [API_REFERENCE.md](API_REFERENCE.md) | Complete API reference for all public classes and functions |
| [SELF_IMPROVEMENT.md](SELF_IMPROVEMENT.md) | Self-improvement loop: 8 phases, patch proposer, reviewer voting, regression detection |

## Quick Reference

| Document | Description |
|----------|-------------|
| [ADAPTERS.md](ADAPTERS.md) | Quick overview of the adapter layer (HTTP, Subprocess, LearningAgent) |
| [LEVELS.md](LEVELS.md) | Quick reference for progressive evaluation levels L1--L16 |
| [MULTI_AGENT_EVAL.md](MULTI_AGENT_EVAL.md) | Multi-agent evaluation pipeline overview |

## Getting Started

1. **Understand the architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md) for the big picture
2. **Choose an adapter**: See [ADAPTERS.md](ADAPTERS.md) or [EXTENDING.md](EXTENDING.md) to connect your agent
3. **Run an evaluation**: Follow the CLI or programmatic examples in [LONG_HORIZON_EVAL.md](LONG_HORIZON_EVAL.md)
4. **Interpret results**: See the "How to Interpret Results" section in [LONG_HORIZON_EVAL.md](LONG_HORIZON_EVAL.md)
5. **Improve your agent**: Use the [Self-Improvement Loop](SELF_IMPROVEMENT.md) for automated improvement
6. **Extend the framework**: See [EXTENDING.md](EXTENDING.md) for adding new levels, categories, and grading dimensions
