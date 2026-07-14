# AgentOps: Productionising AI Agents with Evaluations, Guardrails, Observability & Monitoring

Moving an AI agent from a Jupyter notebook to production requires more than just wrapping it in an API. AgentOps is the discipline of **operationalising** autonomous agents — making them reliable, observable, safe, and measurable in production environments.

## What is AgentOps?

AgentOps covers four pillars that every production agent system needs:

| Pillar | What It Addresses | Why It Matters |
|--------|-------------------|----------------|
| **Evaluations** | How do we measure agent quality? | Without metrics, you can't improve or regress-test |
| **Guardrails** | How do we prevent harmful/incorrect actions? | Agents with tool access can cause real damage |
| **Observability** | What is the agent doing and why? | Debugging agent loops without traces is impossible |
| **Monitoring** | Is the system healthy right now? | Real-time alerts when agents degrade or fail |

### The AgentOps Architecture

```
                         ┌─────────────────────────┐
                         │    User / Application    │
                         └───────────┬─────────────┘
                                     │
                         ┌───────────▼─────────────┐
                         │   Input Guardrails       │
                         │   • Content filtering    │
                         │   • Prompt injection     │
                         │   • Input validation     │
                         └───────────┬─────────────┘
                                     │
                         ┌───────────▼─────────────┐
                         │   Agent Orchestrator     │
                         │   (LangChain, AutoGen,   │
                         │    LangGraph)            │
                         └───────────┬─────────────┘
                                     │
               ┌─────────────────────┼─────────────────────┐
               │                     │                     │
    ┌──────────▼──────────┐ ┌───────▼────────┐ ┌──────────▼──────────┐
    │   LLM Calls         │ │   Tool Calls   │ │   Memory/Context    │
    │   • Tokens logged   │ │   • Success    │ │   • Retrieval       │
    │   • Latency traced  │ │   • Errors     │ │   • Storage         │
    │   • Output validated│ │   • Duration   │ │   • Size tracked    │
    └──────────┬──────────┘ └───────┬────────┘ └──────────┬──────────┘
               │                     │                     │
               └─────────────────────┼─────────────────────┘
                                     │
                         ┌───────────▼─────────────┐
                         │   Output Guardrails      │
                         │   • Hallucination check  │
                         │   • Factual grounding    │
                         │   • Format validation    │
                         └───────────┬─────────────┘
                                     │
                         ┌───────────▼─────────────┐
                         │   Observability Layer    │
                         │   • Traces (OpenTelemetry)│
                         │   • Metrics (Prometheus) │
                         │   • Logs (structured)   │
                         └───────────┬─────────────┘
                                     │
                         ┌───────────▼─────────────┐
                         │   Monitoring & Alerts    │
                         │   • Dashboard (Grafana) │
                         │   • Anomaly detection   │
                         │   • Drift monitoring    │
                         └─────────────────────────┘
```

---

## 1. Evaluations: Measuring Agent Quality

Without evaluations, you're flying blind. Agents can silently degrade as LLMs change, prompts drift, or tools update their APIs.

### Types of Agent Evaluations

| Evaluation Type | What It Tests | How It Works |
|----------------|---------------|--------------|
| **Task Completion** | Did the agent achieve the goal? | Human or LLM judge scores the final output |
| **Step Accuracy** | Did each tool call have correct arguments? | Compare actual tool calls to expected ground truth |
| **Tool Selection** | Did the agent pick the right tool? | Precision/recall against labelled decisions |
| **Latency** | How fast did the agent respond? | Measure end-to-end and per-step timing |
| **Token Efficiency** | How many tokens were used? | Count input/output tokens per task |
| **Hallucination Rate** | Did the agent fabricate facts? | Embedding similarity + consistency checks |
| **Safety Score** | Did the agent follow guardrails? | Automated rule checking on outputs |

### Building an Evaluation Pipeline

```python
from typing import Any
import json
import time
from dataclasses import dataclass, field

@dataclass
class AgentStep:
    """A single step in an agent's execution trace."""
    timestamp: float
    step_type: str  # "llm_call", "tool_call", "tool_result", "final_answer"
    input: str | dict | None = None
    output: str | dict | None = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    score: float | None = None

@dataclass
class AgentEpisode:
    """A complete agent run from query to final answer."""
    query: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    ground_truth: str | None = None
    task_completed: bool = False
    hallucination_score: float = 0.0
    total_duration_ms: float = 0.0
    total_tokens: int = 0

    def add_step(self, step: AgentStep):
        self.steps.append(step)
        self.total_duration_ms += step.duration_ms
        self.total_tokens += step.tokens_used


class AgentEvaluator:
    """Evaluates agent performance on a test dataset."""

    def __init__(self, judge_llm=None):
        self.judge_llm = judge_llm  # LLM-as-judge for qualitative scores

    def evaluate_completion(self, episode: AgentEpisode) -> dict:
        """Evaluate if the agent completed the task correctly."""
        scores = {}

        # 1. Exact match (if ground truth available)
        if episode.ground_truth:
            scores["exact_match"] = float(
                episode.final_answer.strip() == episode.ground_truth.strip()
            )

        # 2. LLM-as-judge evaluation
        if self.judge_llm and episode.ground_truth:
            judge_prompt = f"""
            Rate the answer on a scale of 0-1:
            Query: {episode.query}
            Expected: {episode.ground_truth}
            Got: {episode.final_answer}
            Score based on correctness and completeness.
            Return only a number between 0 and 1.
            """
            response = self.judge_llm.invoke(judge_prompt)
            try:
                scores["llm_judge_score"] = float(response.content.strip())
            except ValueError:
                scores["llm_judge_score"] = 0.0

        # 3. Tool call accuracy
        correct_tools = 0
        for step in episode.steps:
            if step.step_type == "tool_call" and step.score is not None:
                correct_tools += 1 if step.score >= 0.5 else 0
        total_tools = sum(1 for s in episode.steps if s.step_type == "tool_call")
        scores["tool_accuracy"] = correct_tools / max(total_tools, 1)

        # 4. Efficiency metrics
        scores["latency_seconds"] = episode.total_duration_ms / 1000
        scores["total_tokens"] = episode.total_tokens
        scores["num_steps"] = len(episode.steps)
        scores["task_completed"] = episode.task_completed
        scores["hallucination_score"] = episode.hallucination_score

        return scores

    def run_evaluation_suite(self, episodes: list[AgentEpisode]) -> dict:
        """Run evaluations across a dataset of episodes."""
        all_scores = [self.evaluate_completion(ep) for ep in episodes]

        # Aggregate metrics
        metrics = {
            "avg_llm_judge_score": sum(s.get("llm_judge_score", 0) for s in all_scores) / max(len(all_scores), 1),
            "avg_tool_accuracy": sum(s["tool_accuracy"] for s in all_scores) / max(len(all_scores), 1),
            "avg_latency_seconds": sum(s["latency_seconds"] for s in all_scores) / max(len(all_scores), 1),
            "avg_tokens_per_task": sum(s["total_tokens"] for s in all_scores) / max(len(all_scores), 1),
            "completion_rate": sum(s["task_completed"] for s in all_scores) / max(len(all_scores), 1),
            "avg_hallucination_score": sum(s["hallucination_score"] for s in all_scores) / max(len(all_scores), 1),
            "total_episodes": len(episodes),
            "total_tokens": sum(s["total_tokens"] for s in all_scores),
            "p50_latency": sorted(s["latency_seconds"] for s in all_scores)[len(all_scores) // 2],
            "p95_latency": sorted(s["latency_seconds"] for s in all_scores)[int(len(all_scores) * 0.95)],
        }
        return metrics
```

### Running Evaluations in CI/CD

```yaml
# .github/workflows/agent-eval.yml
name: Agent Evaluation Suite
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run evaluation suite
        run: |
          python -m pytest tests/evaluations/ \
            --eval-dataset=test_data/agent_tasks.json \
            --output=eval_results.json

      - name: Check quality gates
        run: |
          python scripts/check_quality_gates.py \
            --results=eval_results.json \
            --min-llm-score=0.75 \
            --max-hallucination=0.15 \
            --max-latency=30.0

      - name: Upload evaluation results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: eval_results.json
```

---

## 2. Guardrails: Keeping Agents Safe

Guardrails are the safety barriers around an agent. They prevent harmful inputs from reaching the LLM and block dangerous outputs from being executed.

### Guardrail Architecture

```
Input → ┌─────────────────┐ → Agent → ┌──────────────────┐ → Output
        │ Input Guardrails │           │ Output Guardrails │
        │ • Toxicity check │           │ • Hallucination   │
        │ • PII detection  │           │   detection       │
        │ • Prompt inject  │           │ • Factual check   │
        │ • Rate limit     │           │ • Format validate │
        │ • Topic allowlist│           │ • Safety filter   │
        └─────────────────┘           └──────────────────┘
```

### Input Guardrails

```python
import re
from typing import Callable

class InputGuardrail:
    """Base class for input guardrails."""
    def check(self, text: str) -> tuple[bool, str]:
        """Returns (passes: bool, reason: str)."""
        raise NotImplementedError

class ToxicityGuardrail(InputGuardrail):
    """Block toxic or abusive inputs."""
    def __init__(self):
        # In production: use a dedicated toxicity model (e.g., Llama Guard)
        self.toxic_patterns = [
            r"ignore all previous instructions",
            r"you are now",
            r"system prompt",
        ]

    def check(self, text: str) -> tuple[bool, str]:
        for pattern in self.toxic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Input blocked: matched pattern '{pattern}'"
        return True, ""

class PIIGuardrail(InputGuardrail):
    """Detect and block personally identifiable information."""
    def __init__(self):
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
        }

    def check(self, text: str) -> tuple[bool, str]:
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                return False, f"Input blocked: contains PII ({pii_type})"
        return True, ""

class TopicGuardrail(InputGuardrail):
    """Only allow queries about allowed topics."""
    def __init__(self, allowed_topics: list[str]):
        self.allowed = allowed_topics

    def check(self, text: str) -> tuple[bool, str]:
        # In production: use an LLM or embedding classifier
        text_lower = text.lower()
        for topic in self.allowed:
            if topic.lower() in text_lower:
                return True, ""
        return False, f"Input blocked: topic not in allowed list ({self.allowed})"


class GuardrailPipeline:
    """Run multiple guardrails sequentially."""
    def __init__(self, guardrails: list[InputGuardrail]):
        self.guardrails = guardrails

    def check(self, text: str) -> tuple[bool, list[str]]:
        all_reasons = []
        for guardrail in self.guardrails:
            passes, reason = guardrail.check(text)
            if not passes:
                all_reasons.append(reason)
        return len(all_reasons) == 0, all_reasons


# Usage
pipeline = GuardrailPipeline([
    ToxicityGuardrail(),
    PIIGuardrail(),
    TopicGuardrail(allowed_topics=["finance", "stocks", "investing"]),
])

text = "Tell me about Apple's stock"
passes, reasons = pipeline.check(text)
print(f"Passes: {passes}, Reasons: {reasons if not passes else 'None'}")
```

### Output Guardrails (with Hallucination Detection)

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HallucinationDetector:
    """Detect hallucinations by comparing output to evidence."""

    def __init__(self, threshold: float = 0.72):
        # In production: use the same embedding model as your RAG pipeline
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

    def compute_grounding_score(self, answer: str, evidence: list[str]) -> float:
        """Compute how well the answer is grounded in evidence."""
        if not evidence:
            return 0.0

        answer_emb = self.model.encode(answer)
        evidence_embs = self.model.encode(evidence)

        # Max similarity to any evidence chunk
        similarities = cosine_similarity([answer_emb], evidence_embs)[0]
        return float(np.max(similarities))

    def check(self, answer: str, evidence: list[str]) -> tuple[bool, float, str]:
        """
        Check if the answer is grounded in evidence.
        Returns (passes, score, message).
        """
        score = self.compute_grounding_score(answer, evidence)
        if score >= self.threshold:
            return True, score, f"✅ Grounded (score={score:.3f})"
        else:
            return False, score, f"⚠️ Possible hallucination (score={score:.3f})"


class ConsistencyChecker:
    """Check if LLM output contradicts itself across multiple calls."""

    def check_consistency(self, responses: list[str]) -> float:
        """Measure internal consistency of multiple responses."""
        if len(responses) < 2:
            return 1.0

        embs = self.model.encode(responses)
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = cosine_similarity([embs[i]], [embs[j]])[0][0]
                similarities.append(sim)

        # Low consistency = high hallucination risk
        return float(np.mean(similarities))


class OutputGuardrailPipeline:
    """Run output guardrails on agent responses."""

    def __init__(self, hallucination_threshold: float = 0.72):
        self.hallucination_detector = HallucinationDetector(threshold=hallucination_threshold)
        self.consistency_checker = ConsistencyChecker()

    def check(
        self, answer: str, evidence: list[str], previous_responses: list[str] = None
    ) -> dict:
        results = {}

        # Hallucination check
        grounded, score, msg = self.hallucination_detector.check(answer, evidence)
        results["hallucination_check"] = {
            "passes": grounded,
            "score": score,
            "message": msg,
        }

        # Consistency check
        if previous_responses:
            consistency = self.consistency_checker.check_consistency(
                previous_responses + [answer]
            )
            results["consistency_check"] = {
                "score": consistency,
                "passes": consistency > 0.65,
                "message": f"Consistency: {consistency:.3f}",
            }

        # Overall pass/fail
        results["passes"] = all(
            check.get("passes", True) for check in results.values()
        )
        return results
```

---

## 3. Observability: Traces, Logs, and Metrics

Observability is about understanding what your agent is doing in production. The three pillars are **traces**, **logs**, and **metrics**.

### Tracing with OpenTelemetry

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
import time
import uuid

# Setup OpenTelemetry tracer
resource = Resource.create({"service.name": "financial-agent"})
provider = TracerProvider(resource=resource)

# Export to your observability backend (Jaeger, Grafana Tempo, etc.)
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317", insecure=True
)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)


class ObservableAgent:
    """Agent wrapper with OpenTelemetry tracing."""

    def invoke(self, query: str) -> str:
        episode_id = str(uuid.uuid4())[:8]

        with tracer.start_as_current_span(f"agent_run_{episode_id}") as root_span:
            root_span.set_attribute("query", query)
            root_span.set_attribute("agent.type", "financial_analyst")

            # Input guardrails
            with tracer.start_as_current_span("input_guardrails") as span:
                passes, reasons = guardrail_pipeline.check(query)
                span.set_attribute("passes", passes)
                span.set_attribute("reasons", str(reasons))
                if not passes:
                    root_span.set_attribute("blocked", True)
                    return "Query blocked by guardrails."

            # LLM call with tracing
            with tracer.start_as_current_span("llm_call") as span:
                start = time.time()
                response = llm.invoke(query)
                duration = (time.time() - start) * 1000
                span.set_attribute("duration_ms", duration)
                span.set_attribute("tokens_used", estimate_tokens(query) + estimate_tokens(response.content))
                span.set_attribute("model", "llama3.1")

            # Tool calls
            for tool_call in extract_tool_calls(response):
                with tracer.start_as_current_span(f"tool_{tool_call['name']}") as span:
                    span.set_attribute("tool.name", tool_call["name"])
                    span.set_attribute("tool.args", str(tool_call["args"]))
                    start = time.time()
                    result = execute_tool(tool_call)
                    span.set_attribute("tool.duration_ms", (time.time() - start) * 1000)
                    span.set_attribute("tool.success", result["success"])

            # Output guardrails
            with tracer.start_as_current_span("output_guardrails") as span:
                guardrail_results = output_pipeline.check(
                    answer=final_answer,
                    evidence=retrieved_evidence,
                )
                span.set_attribute("hallucination_score", guardrail_results["hallucination_check"]["score"])
                span.set_attribute("passes_guardrails", guardrail_results["passes"])

            return final_answer
```

### Structured Logging

```python
import structlog
import json
from datetime import datetime

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if __debug__
        else structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


class LoggedAgent:
    """Agent with structured logging for every step."""

    def invoke(self, query: str) -> str:
        episode_id = str(uuid.uuid4())[:8]
        logger.info("agent.start", episode_id=episode_id, query=query, query_length=len(query))

        # Input guardrail check
        passes, reasons = pipeline.check(query)
        logger.info("agent.input_guardrails",
                    episode_id=episode_id, passes=passes, reasons=reasons)

        if not passes:
            logger.warning("agent.blocked", episode_id=episode_id, reasons=reasons)
            return "Blocked."

        # LLM call
        logger.info("agent.llm_call.start", episode_id=episode_id)
        start = time.time()
        response = llm.invoke(query)
        duration = (time.time() - start) * 1000
        logger.info("agent.llm_call.complete",
                    episode_id=episode_id, duration_ms=duration,
                    response_length=len(response.content))

        # Tool calls
        for tc in extract_tool_calls(response):
            logger.info("agent.tool_call.start",
                       episode_id=episode_id, tool=tc["name"], args=tc["args"])
            start = time.time()
            result = execute_tool(tc)
            logger.info("agent.tool_call.complete",
                       episode_id=episode_id, tool=tc["name"],
                       duration_ms=(time.time() - start) * 1000, success=result["success"])

        # Output guardrails
        gr = output_pipeline.check(answer=final_answer, evidence=evidence)
        logger.info("agent.output_guardrails",
                   episode_id=episode_id, passes=gr["passes"],
                   hallucination_score=gr["hallucination_check"]["score"])

        logger.info("agent.complete", episode_id=episode_id)
        return final_answer
```

### Metrics (Prometheus + Grafana)

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
AGENT_REQUESTS = Counter("agent_requests_total", "Total agent requests", ["status"])
AGENT_LATENCY = Histogram(
    "agent_latency_seconds",
    "Agent request latency",
    buckets=[1, 2.5, 5, 10, 15, 20, 30, 60],
)
TOOL_CALLS = Counter("tool_calls_total", "Tool calls", ["tool", "status"])
LLM_TOKENS = Counter("llm_tokens_total", "Tokens used", ["type"])  # type: input/output
HALLUCINATION_SCORE = Gauge("hallucination_score", "Current hallucination score")
GUARDRAIL_BLOCKS = Counter("guardrail_blocks_total", "Blocked requests", ["guardrail"])
ACTIVE_SESSIONS = Gauge("active_agent_sessions", "Currently active agent sessions")


class MetricsAgent:
    """Agent wrapper with Prometheus metrics."""

    def invoke(self, query: str) -> str:
        ACTIVE_SESSIONS.inc()
        start = time.time()

        try:
            # Input guardrails
            passes, reasons = pipeline.check(query)
            if not passes:
                GUARDRAIL_BLOCKS.labels(guardrail="input").inc()
                AGENT_REQUESTS.labels(status="blocked").inc()
                return "Blocked."

            # LLM call
            response = llm.invoke(query)
            LLM_TOKENS.labels(type="input").inc(len(query.split()))
            LLM_TOKENS.labels(type="output").inc(len(response.content.split()))

            # Tool calls
            for tc in extract_tool_calls(response):
                try:
                    result = execute_tool(tc)
                    TOOL_CALLS.labels(tool=tc["name"], status="success").inc()
                except Exception as e:
                    TOOL_CALLS.labels(tool=tc["name"], status="error").inc()

            # Output guardrails
            gr = output_pipeline.check(answer=final_answer, evidence=evidence)
            HALLUCINATION_SCORE.set(gr["hallucination_check"]["score"])

            if not gr["passes"]:
                GUARDRAIL_BLOCKS.labels(guardrail="output").inc()
                AGENT_REQUESTS.labels(status="hallucination").inc()
                return "Answer blocked due to hallucination risk."

            AGENT_REQUESTS.labels(status="success").inc()
            return final_answer

        finally:
            duration = time.time() - start
            AGENT_LATENCY.observe(duration)
            ACTIVE_SESSIONS.dec()


# Start metrics server
start_http_server(8000)
# Metrics available at http://localhost:8000/metrics
```

---

## 4. Monitoring: Real-Time Dashboards and Alerts

Monitoring turns observability data into actionable insights. Key metrics to monitor:

| Metric | What It Tells You | Alert Threshold |
|--------|-------------------|-----------------|
| Error rate | % of agent runs that fail | >5% over 5 minutes |
| Hallucination score | Average grounding score | <0.72 over 10 runs |
| Latency P95 | Slowest agent responses | >30 seconds |
| Token usage | Cost per query | >$0.01 per query |
| Guardrail blocks | How often safety catches fire | >10% of requests |
| Tool error rate | External API health | >5% errors on any tool |
| Session duration | Agent loop length | >60 seconds |

### Grafana Dashboard PromQL Queries

```promql
# Error rate over 5 minutes
rate(agent_requests_total{status=~"error|blocked|hallucination"}[5m])
/
rate(agent_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(agent_latency_seconds_bucket[5m]))

# Hallucination score trend
avg_over_time(hallucination_score[15m])

# Tool error rate by tool
rate(tool_calls_total{status="error"}[5m])
/
rate(tool_calls_total[5m])

# Token cost per hour
rate(llm_tokens_total[1h]) * 0.00000015  # $0.15 per 1M tokens
```

### Alert Rules

```yaml
# alerts.yml
groups:
  - name: agent_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(agent_requests_total{status=~"error|blocked"}[5m]) / rate(agent_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Agent error rate above 5%"

      - alert: HighHallucinationRate
        expr: avg_over_time(hallucination_score[15m]) < 0.72
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Hallucination score below threshold"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(agent_latency_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency above 30 seconds"

      - alert: ToolFailure
        expr: rate(tool_calls_total{status="error"}[5m]) > 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Tool calls failing for {{ $labels.tool }}"
```

---

## 5. The "Lifeguard" Pattern: Catching Hallucinations in Real Time

The **Lifeguard** pattern is a multi-layered approach to detecting and preventing hallucinations as they happen, rather than after the fact. Like a lifeguard watching swimmers, it constantly scans for signs of distress in the agent's outputs.

### Lifeguard Detection Layers

```
                    ┌─────────────────────────────────┐
                    │        Lifeguard Monitor         │
                    ├─────────────────────────────────┤
                    │ Layer 1: Semantic Grounding     │
                    │   • Embedding similarity check  │
                    │   • Evidence retrieval overlap  │
                    ├─────────────────────────────────┤
                    │ Layer 2: Self-Consistency       │
                    │   • Multiple response sampling  │
                    │   • Cross-response agreement    │
                    ├─────────────────────────────────┤
                    │ Layer 3: Factual Verification   │
                    │   • LLM-as-judge cross-exam     │
                    │   • Named entity validation     │
                    ├─────────────────────────────────┤
                    │ Layer 4: Behavioral             │
                    │   • Token probability analysis  │
                    │   • Response pattern anomalies  │
                    └─────────────────────────────────┘
```

### Layer 1: Semantic Grounding

Checks if the LLM output is semantically close to the retrieved evidence.

```python
def lifeguard_grounding_check(
    answer: str,
    evidence_chunks: list[str],
    threshold: float = 0.72,
) -> dict:
    """
    Lifeguard Layer 1: Semantic grounding check.
    Returns a risk assessment and the grounding score.
    """
    answer_emb = embedding_model.encode(answer)
    evidence_embs = embedding_model.encode(evidence_chunks)

    # Best match score
    max_similarity = float(np.max(cosine_similarity([answer_emb], evidence_embs)))

    # Average similarity across all evidence
    avg_similarity = float(np.mean(cosine_similarity([answer_emb], evidence_embs)))

    risk_level = "low" if max_similarity >= threshold else "medium" if max_similarity >= threshold * 0.8 else "high"

    return {
        "risk_level": risk_level,
        "max_similarity": max_similarity,
        "avg_similarity": avg_similarity,
        "passes": max_similarity >= threshold,
    }
```

### Layer 2: Self-Consistency (N-shot Sampling)

Generate the same answer multiple times with different temperatures and check for agreement. High disagreement = high hallucination risk.

```python
def lifeguard_consistency_check(
    llm,
    prompt: str,
    n_samples: int = 3,
    temperature_range: tuple[float, float] = (0.0, 0.5),
) -> dict:
    """
    Lifeguard Layer 2: Self-consistency check.
    Generate multiple responses and measure agreement.
    """
    responses = []
    for i in range(n_samples):
        temp = temperature_range[0] + (temperature_range[1] - temperature_range[0]) * (i / max(n_samples - 1, 1))
        response = llm.invoke(prompt, temperature=temp)
        responses.append(response.content)

    # Check pairwise similarity
    embs = embedding_model.encode(responses)
    pairwise_similarities = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            sim = cosine_similarity([embs[i]], [embs[j]])[0][0]
            pairwise_similarities.append(sim)

    mean_agreement = float(np.mean(pairwise_similarities)) if pairwise_similarities else 1.0
    std_agreement = float(np.std(pairwise_similarities)) if pairwise_similarities else 0.0

    risk_level = "low" if mean_agreement >= 0.8 else "medium" if mean_agreement >= 0.6 else "high"

    return {
        "risk_level": risk_level,
        "mean_agreement": mean_agreement,
        "std_agreement": std_agreement,
        "responses": responses,
        "passes": mean_agreement >= 0.65,
    }
```

### Layer 3: Factual Verification (LLM Cross-Examination)

Use a second LLM to verify specific claims made by the first LLM.

```python
def lifeguard_factual_verify(
    answer: str,
    context: str,
    verifier_llm,
) -> dict:
    """
    Lifeguard Layer 3: Factual verification via LLM cross-examination.
    """
    prompt = f"""You are a fact-checker. Verify each claim in the answer against the context.

Context:
{context}

Answer to verify:
{answer}

For each claim in the answer, state:
- CLAIM: the specific claim
- STATUS: SUPPORTED / CONTRADICTED / NOT_IN_CONTEXT
- EVIDENCE: the supporting or contradicting text from context

Then give an overall verdict: PASS or FAIL.
"""
    verification = verifier_llm.invoke(prompt)
    content = verification.content.strip()

    passes = "PASS" in content and "FAIL" not in content[:content.find("PASS") + 5] if "PASS" in content else False
    has_contradictions = "CONTRADICTED" in content

    risk_level = "low" if passes else "high" if has_contradictions else "medium"

    return {
        "risk_level": risk_level,
        "passes": passes,
        "has_contradictions": has_contradictions,
        "verification_text": content,
    }
```

### Layer 4: Behavioral Anomaly Detection

Monitor token-level probability patterns for anomalies. Low-confidence tokens or unexpected entropy shifts can indicate hallucination.

```python
def lifeguard_behavioral_check(
    llm_response,
    entropy_threshold: float = 2.5,
) -> dict:
    """
    Lifeguard Layer 4: Behavioral anomaly detection.
    Uses token log probabilities to detect uncertainty.
    """
    # In production, access token logprobs from the LLM response
    # This is a simplified version
    if not hasattr(llm_response, "response_metadata"):
        return {"risk_level": "unknown", "passes": True}

    # Extract log probabilities if available
    token_probs = []
    for token_data in llm_response.response_metadata.get("logprobs", []):
        if token_data:
            probs = np.exp(token_data.get("logprob", 0))
            token_probs.append(probs)

    if not token_probs:
        return {"risk_level": "unknown", "passes": True}

    mean_prob = float(np.mean(token_probs))
    min_prob = float(np.min(token_probs))

    risk_level = "low" if mean_prob >= 0.7 else "medium" if mean_prob >= 0.4 else "high"

    return {
        "risk_level": risk_level,
        "mean_token_probability": mean_prob,
        "min_token_probability": min_prob,
        "passes": mean_prob >= 0.4,
    }
```

### Full Lifeguard Pipeline

```python
class LifeguardMonitor:
    """
    Multi-layer hallucination detection and prevention.
    Acts as a final output guardrail before the response is sent.
    """

    def __init__(self, embedding_model, verifier_llm, grounding_threshold: float = 0.72):
        self.embedding_model = embedding_model
        self.verifier_llm = verifier_llm
        self.grounding_threshold = grounding_threshold
        self.logger = structlog.get_logger()

    def monitor(
        self,
        answer: str,
        evidence: list[str],
        original_prompt: str,
        llm_response=None,
    ) -> dict:
        """Run all lifeguard checks and return a consolidated risk assessment."""

        results = {}

        # Layer 1: Semantic grounding
        results["grounding"] = lifeguard_grounding_check(
            answer, evidence, self.grounding_threshold
        )
        self.logger.info("lifeguard.layer1",
                        risk=results["grounding"]["risk_level"],
                        score=results["grounding"]["max_similarity"])

        # Layer 2: Self-consistency (only if evidence is weak)
        if results["grounding"]["risk_level"] in ("medium", "high"):
            results["consistency"] = lifeguard_consistency_check(
                self.verifier_llm, original_prompt, n_samples=3
            )
            self.logger.info("lifeguard.layer2",
                            risk=results["consistency"]["risk_level"],
                            agreement=results["consistency"]["mean_agreement"])
        else:
            results["consistency"] = {"risk_level": "low", "passes": True}

        # Layer 3: Factual verification
        results["factual"] = lifeguard_factual_verify(
            answer, "\n".join(evidence[:3]), self.verifier_llm
        )
        self.logger.info("lifeguard.layer3",
                        risk=results["factual"]["risk_level"],
                        passes=results["factual"]["passes"])

        # Layer 4: Behavioral (if logprobs available)
        if llm_response:
            results["behavioral"] = lifeguard_behavioral_check(llm_response)
            self.logger.info("lifeguard.layer4",
                            risk=results["behavioral"]["risk_level"])

        # Overall assessment
        all_risks = [
            r.get("risk_level", "unknown")
            for r in results.values()
            if isinstance(r, dict)
        ]

        if "high" in all_risks:
            overall_risk = "high"
        elif "medium" in all_risks:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        results["overall_risk"] = overall_risk
        results["passes"] = overall_risk != "high"

        if not results["passes"]:
            self.logger.warning("lifeguard.blocked",
                              overall_risk=overall_risk,
                              answer_preview=answer[:100])

        return results
```

---

## 6. Putting It All Together: Production Agent Architecture

```python
class ProductionAgent:
    """
    Fully productionised agent with evaluations, guardrails,
    observability, monitoring, and lifeguard hallucination detection.
    """

    def __init__(self, llm, tools, embedding_model, verifier_llm):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.lifeguard = LifeguardMonitor(embedding_model, verifier_llm)
        self.logger = structlog.get_logger()

        # Guardrails
        self.input_guardrails = GuardrailPipeline([
            ToxicityGuardrail(),
            PIIGuardrail(),
            TopicGuardrail(allowed_topics=["finance", "stocks", "investing", "economy"]),
        ])
        self.output_guardrails = OutputGuardrailPipeline()

        # Metrics
        self.metrics = {
            "requests": AGENT_REQUESTS,
            "latency": AGENT_LATENCY,
            "tool_calls": TOOL_CALLS,
            "hallucination": HALLUCINATION_SCORE,
            "guardrail_blocks": GUARDRAIL_BLOCKS,
        }

        # Tracer
        self.tracer = trace.get_tracer(__name__)

    def invoke(self, query: str) -> dict:
        episode_id = str(uuid.uuid4())[:8]

        with self.tracer.start_as_current_span(f"agent_{episode_id}") as span:
            span.set_attribute("query", query)
            self.logger.info("agent.start", episode_id=episode_id)

            # Step 1: Input guardrails
            passes, reasons = self.input_guardrails.check(query)
            if not passes:
                GUARDRAIL_BLOCKS.labels(guardrail="input").inc()
                AGENT_REQUESTS.labels(status="blocked").inc()
                self.logger.warning("agent.blocked", reasons=reasons)
                return {"episode_id": episode_id, "status": "blocked", "response": None}

            # Step 2: Agent loop with tracing
            steps = []
            with self.tracer.start_as_current_span("agent_loop") as loop_span:
                messages = [{"role": "user", "content": query}]
                final_answer = None

                for iteration in range(5):  # max iterations
                    with self.tracer.start_as_current_span(f"iteration_{iteration}"):
                        # LLM call
                        start = time.time()
                        response = self.llm.invoke(messages)
                        llm_duration = (time.time() - start) * 1000

                        LLM_TOKENS.labels(type="input").inc(len(str(messages).split()))
                        LLM_TOKENS.labels(type="output").inc(len(response.content.split()))

                        # Check for tool calls
                        if hasattr(response, "tool_calls") and response.tool_calls:
                            for tc in response.tool_calls:
                                tool_name = tc["name"]
                                if tool_name in self.tools:
                                    with self.tracer.start_as_current_span(f"tool_{tool_name}"):
                                        start = time.time()
                                        try:
                                            result = self.tools[tool_name].invoke(tc["args"])
                                            TOOL_CALLS.labels(tool=tool_name, status="success").inc()
                                        except Exception as e:
                                            result = f"Error: {e}"
                                            TOOL_CALLS.labels(tool=tool_name, status="error").inc()

                                        messages.append({
                                            "role": "tool",
                                            "content": str(result),
                                            "tool_call_id": tc["id"],
                                        })
                        else:
                            final_answer = response.content
                            break

            # Step 3: Output guardrails + Lifeguard
            if final_answer:
                with self.tracer.start_as_current_span("lifeguard_check"):
                    lifeguard_results = self.lifeguard.monitor(
                        answer=final_answer,
                        evidence=[m["content"] for m in messages if m["role"] == "tool"],
                        original_prompt=query,
                        llm_response=response,
                    )

                    HALLUCINATION_SCORE.set(
                        lifeguard_results.get("grounding", {}).get("max_similarity", 0)
                    )

                    if not lifeguard_results["passes"]:
                        GUARDRAIL_BLOCKS.labels(guardrail="output").inc()
                        AGENT_REQUESTS.labels(status="hallucination").inc()
                        self.logger.warning("agent.hallucination_detected",
                                          overall_risk=lifeguard_results["overall_risk"])
                        return {
                            "episode_id": episode_id,
                            "status": "hallucination_risk",
                            "response": None,
                            "lifeguard_results": lifeguard_results,
                        }

            # Step 4: Record success
            AGENT_REQUESTS.labels(status="success").inc()
            self.logger.info("agent.complete", episode_id=episode_id)

            return {
                "episode_id": episode_id,
                "status": "success",
                "response": final_answer,
                "steps": len(steps),
            }
```

---

## 7. Production Checklist

| Area | Item | Priority |
|------|------|----------|
| **Evaluations** | Create a test dataset of 100+ queries with ground truth | 🔴 Critical |
| **Evaluations** | Implement LLM-as-judge automated scoring | 🔴 Critical |
| **Evaluations** | Set up CI/CD pipeline that runs evals on every commit | 🟡 High |
| **Evaluations** | Define quality gates (min score, max hallucination, max latency) | 🟡 High |
| **Guardrails** | Implement input guardrails (toxicity, PII, prompt injection) | 🔴 Critical |
| **Guardrails** | Implement output guardrails (hallucination, factual, format) | 🔴 Critical |
| **Guardrails** | Set up rate limiting and request throttling | 🟡 High |
| **Observability** | Add OpenTelemetry tracing to every agent step | 🔴 Critical |
| **Observability** | Set up structured logging with episode IDs | 🔴 Critical |
| **Observability** | Export metrics (Prometheus format) | 🔴 Critical |
| **Monitoring** | Create Grafana dashboard for all key metrics | 🟡 High |
| **Monitoring** | Set up alerts for error rate, latency, hallucination | 🟡 High |
| **Hallucination** | Implement Lifeguard multi-layer detection | 🔴 Critical |
| **Hallucination** | Log all hallucination events for post-mortem analysis | 🟡 High |
| **Cost** | Track token usage per user/query | 🟢 Nice-to-have |

---

## Files

- `autogen_ops.ipynb`: Jupyter notebook with runnable examples for all concepts above
- `agent_eval.py`: Reusable evaluation framework
- `lifeguard.py`: Multi-layer hallucination detection module

## Suggested Flow

1. Start with **Evaluations** — you can't improve what you can't measure.
2. Add **Guardrails** — prevent bad outputs before they reach users.
3. Instrument **Observability** — traces, logs, and metrics for every step.
4. Set up **Monitoring** — real-time dashboards and alerts.
5. Implement **Lifeguard** — multi-layer hallucination detection as your safety net.
6. Continuously run evaluations in CI/CD to catch regressions.