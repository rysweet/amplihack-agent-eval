# Writing Custom Agent Adapters

## The AgentAdapter Interface

To make any agent evaluable, implement the `AgentAdapter` abstract base class from `amplihack_eval.adapters.base`:

```python
from amplihack_eval import AgentAdapter, AgentResponse, ToolCall

class MyAgent(AgentAdapter):
    def learn(self, content: str) -> None:
        """Feed content to the agent for learning/memorization."""
        ...

    def answer(self, question: str) -> AgentResponse:
        """Ask the agent a question. Returns answer + trajectory."""
        ...

    def reset(self) -> None:
        """Reset agent state between eval runs."""
        ...

    def close(self) -> None:
        """Clean up resources (connections, files, etc.)."""
        ...
```

### Required Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `learn` | `(content: str) -> None` | Feed content to the agent. Called once per dialogue turn during evaluation. |
| `answer` | `(question: str) -> AgentResponse` | Ask the agent a question. Must return an `AgentResponse`. |
| `reset` | `() -> None` | Reset all agent state. Called between evaluation runs. |
| `close` | `() -> None` | Clean up resources. Called when evaluation is complete. |

### Optional Properties

| Property | Type | Default | Purpose |
|----------|------|---------|---------|
| `capabilities` | `set[str]` | `{"memory"}` | Declare what the agent can do. Used by the runner to select appropriate eval levels. |
| `name` | `str` | Class name | Human-readable name for reports and logs. |

### AgentResponse

The `answer()` method must return an `AgentResponse`:

```python
@dataclass
class AgentResponse:
    answer: str                              # Required: the agent's answer text
    tool_calls: list[ToolCall] = []          # Optional: tool invocations
    reasoning_trace: str = ""                # Optional: chain-of-thought
    confidence: float = 0.0                  # Optional: self-reported confidence
    metadata: dict[str, Any] = {}            # Optional: arbitrary metadata
```

### ToolCall

If your agent uses tools, capture them for trajectory analysis:

```python
@dataclass
class ToolCall:
    tool_name: str               # Name of the tool invoked
    arguments: dict[str, Any]    # Arguments passed to the tool
    result: str                  # String result from the tool
    timestamp: float = 0.0       # Optional: when the call happened
```

## Built-in Adapters

### HttpAdapter

For agents exposed via REST API:

```python
from amplihack_eval.adapters.http_adapter import HttpAdapter

adapter = HttpAdapter(
    base_url="http://localhost:8000",
    timeout=30,
)
```

Expected endpoints:
- `POST /learn` with `{"content": "..."}` -> 200 OK
- `POST /answer` with `{"question": "..."}` -> `{"answer": "...", "tool_calls": [...], ...}`
- `POST /reset` -> 200 OK

### SubprocessAdapter

For agents invokable via CLI:

```python
from amplihack_eval.adapters.subprocess_adapter import SubprocessAdapter

adapter = SubprocessAdapter(
    command=["python", "my_agent.py"],
    learn_flag="--learn",
    answer_flag="--answer",
)
```

### LearningAgentAdapter

For the amplihack LearningAgent (requires `amplihack` package):

```python
from amplihack_eval.adapters.learning_agent import LearningAgentAdapter

adapter = LearningAgentAdapter()
```

## Complete Custom Adapter Example

```python
from amplihack_eval import AgentAdapter, AgentResponse, ToolCall

class RAGAgent(AgentAdapter):
    """Adapter for a retrieval-augmented generation agent."""

    def __init__(self, db_url: str, model: str = "gpt-4"):
        self.db_url = db_url
        self.model = model
        self.client = VectorDBClient(db_url)
        self.llm = LLMClient(model)

    def learn(self, content: str) -> None:
        # Chunk and embed content into vector DB
        chunks = self._chunk(content)
        embeddings = self.llm.embed(chunks)
        self.client.upsert(chunks, embeddings)

    def answer(self, question: str) -> AgentResponse:
        # Retrieve relevant chunks
        query_embedding = self.llm.embed([question])[0]
        results = self.client.search(query_embedding, top_k=5)

        # Generate answer with context
        context = "\n".join(r.text for r in results)
        response = self.llm.generate(
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        return AgentResponse(
            answer=response.text,
            tool_calls=[
                ToolCall(
                    tool_name="vector_search",
                    arguments={"query": question, "top_k": 5},
                    result=f"Found {len(results)} chunks",
                )
            ],
            reasoning_trace=f"Retrieved {len(results)} chunks, generated answer",
            confidence=response.confidence,
        )

    def reset(self) -> None:
        self.client.clear()

    def close(self) -> None:
        self.client.close()
        self.llm.close()

    @property
    def capabilities(self) -> set[str]:
        return {"memory", "tool_use"}

    @property
    def name(self) -> str:
        return f"RAGAgent({self.model})"
```

## Running Evaluation with a Custom Adapter

```python
from amplihack_eval import EvalRunner

agent = RAGAgent(db_url="http://localhost:6333", model="gpt-4")
runner = EvalRunner(num_turns=100, num_questions=20, grader_votes=3)
report = runner.run(agent)

print(f"Overall: {report.overall_score:.2%}")
for cb in report.category_breakdown:
    print(f"  {cb.category}: {cb.avg_score:.2%}")

agent.close()
```

## Tips

- **Keep `learn()` fast**: The runner calls it once per dialogue turn (potentially 1000+ times). Batch operations if possible.
- **Capture tool calls**: Even if your agent does not use explicit tools, logging internal retrieval as a `ToolCall` enables richer analysis.
- **Set confidence**: If your agent can estimate confidence, include it. The grader uses confidence calibration in advanced eval levels (L8).
- **Reset completely**: `reset()` must clear ALL state. Leftover state between runs corrupts multi-seed evaluation.
