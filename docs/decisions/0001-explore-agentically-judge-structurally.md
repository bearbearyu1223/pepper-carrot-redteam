# ADR 0001 — Explore agentically, judge structurally

- **Status:** Accepted (skeleton)
- **Date:** 2026-06-07
- **Context:** `pepper-carrot-redteam` (Post 19), the discovery half of evaluation

## Context

`pepper-carrot-eval` (Post 18) is a *deterministic* harness: fixed gold, scripted tool dispatch,
reproducible scores. Its strength is also its ceiling — it only catches failures someone already
wrote a gold case for. We want a tool that *discovers* unknown failures by letting an LLM agent
drive the same `search`/`ask` tools and adapt its probes.

The danger is obvious: if an LLM both *runs the attack* and *decides whether it worked*, the result
is a fluent, unfalsifiable story. An agent is motivated and creative; it is not a trustworthy judge
of its own success, and its verdicts aren't reproducible.

## Decision

**Separate the two roles in code and in trust:**

1. **The agent decides actions only.** It picks queries, chooses `search` vs. `ask`, and adapts
   across turns. This is the part that genuinely benefits from agency.
2. **An oracle decides verdicts, as deterministically as the failure mode allows.**
   - **Structural where possible.** Spoiler success is a checkable predicate on returned positions —
     the *same* `(episode, page)` boundary check the eval uses (`_past_boundary`). We port it
     verbatim so both repos agree on what a "leak" is. No model in the verdict path.
   - **Guarded LLM judge only for genuinely fuzzy modes** (e.g. hallucination), reusing the eval's
     guards: cross-model, anchored rubric, forced structured output, temperature 0. Flagged
     lower-confidence than structural verdicts.
3. **The output is candidate gold, not a score.** Confirmed failures are written in the evaluator's
   gold schema for human review, then frozen into `pepper-carrot-eval`. Discovery feeds measurement;
   it does not replace it.

These live in separate modules (`agent.py` decides, `oracle.py` judges) so the boundary is auditable
— mirroring how the eval keeps dispatch (`run.py`) out of scoring.

## Consequences

- **Credible findings.** A reported failure is backed by a checkable oracle or a guarded judge, not
  the attacker's say-so.
- **Not reproducible, and we say so.** Agentic runs vary; the artifact is coverage (transcripts +
  candidate gold), not a number. The deterministic eval stays the regression gate.
- **Bounded cost.** Because the agent loops and `ask` costs real money, a budget governor (max
  turns / tool calls / USD) is mandatory, not optional.
- **Shared definition of failure.** Porting the eval's boundary predicate (rather than re-deriving
  it) means redteam and eval can never disagree about what counts as a spoiler leak.

## Alternatives considered

- **Let the agent self-score.** Rejected: unfalsifiable, non-reproducible, rewards confident prose.
- **Fold red-teaming into the eval as a "mode."** Rejected: it would put a non-reproducible,
  model-driven path inside the scored loop and contaminate the metrics. Keep them separate repos,
  connected only at the freeze line (candidate gold).
- **Skip structural oracles, use only an LLM judge.** Rejected for spoilers: the boundary *is* a
  structural property, so a structural check is both cheaper and stronger than a judge's opinion.
