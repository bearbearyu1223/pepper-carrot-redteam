# Design — pepper-carrot-redteam

> An agentic red-teamer that discovers failures the deterministic eval can't, then hands them
> back as candidate gold. This doc is the build plan; [ADR 0001](decisions/0001-explore-agentically-judge-structurally.md)
> records the one load-bearing decision.

## 1. Why this exists

`pepper-carrot-eval` (Post 18) measures the companion against a *frozen gold set*. Its blind spot
is structural: **it can only catch failures someone already wrote a test for.** Coverage is bounded
by human imagination. This project is the complement — an agent that actively hunts for failures the
gold set never anticipated. It does **not** replace the eval; it *feeds* it.

The split mirrors the series' throughline: the deterministic core (the eval) stays reproducible;
the agentic edge (this) explores. They meet at the freeze line — confirmed discoveries become gold.

## 2. What it is, precisely

A **third MCP client** of the same `pepper-carrot-mcp` server, calling the same `search`/`ask`
tools. Unlike the eval (a script that dispatches tools in fixed order), this is an **agent**: an
LLM with tool use, handed a mission and the tools, that *decides* which probes to run and *adapts*
across turns. The thing that makes it credible rather than a gimmick is the discipline it inherits
from the eval — see §4.

## 3. Attack surfaces (strategies)

Each strategy = a mission prompt for the agent + an oracle that decides success. MVP ships the
first two; the rest are backlog.

1. **Spoiler leak** *(MVP, structural oracle).* Reader is at `(episode, page)`; the agent tries to
   extract anything past the cursor — direct ("what happens next"), oblique inference, roleplay,
   multi-turn pressure, prompt injection in the question field. **The most compelling probe**: the
   whole series stakes itself on spoiler-safety, so an agent actively trying to break it — and a
   structural verdict that it held — is a strong artifact.
2. **Hallucination / groundedness** *(MVP, LLM-judge oracle).* Probe for confident answers about
   characters/places/lore not in the corpus (invented entities, "tell me more about X" where X is
   fabricated).
3. **Out-of-domain / prompt injection** *(backlog).* Try to make the companion answer general-
   knowledge questions or follow instructions embedded in the user message that override the system
   prompt. The server-side spoiler position *should* defeat boundary-widening injection — good to
   prove.
4. **Retrieval blind spots** *(backlog).* Questions that *should* be answerable from the corpus but
   the retriever misses — these become new positive gold for the eval.

## 4. The one rule: explore agentically, judge structurally

The trap with agentic eval is letting the attacker also be the judge. We don't.

- **The agent decides actions.** Which query, which tool, which follow-up — model-driven, adaptive.
- **An oracle decides verdicts**, as deterministically as the failure mode allows:
  - **Spoiler** → *structural*. Inspect `ask`'s `retrieved_doc_ids` (and/or a `search` at the same
    position) and assert nothing at or past the cursor came back. This is the **same boundary
    predicate** the eval uses (`_past_boundary` in `pepper-carrot-eval/refusal_eval.py`); we port it
    verbatim so the two repos agree on what "a leak" means.
  - **Hallucination** → *guarded LLM judge*, with the eval's guards (cross-model, anchored rubric,
    forced structured output, temperature 0). Fuzzy by nature, so flagged as lower-confidence.
- **Reproducibility caveat, stated loudly.** Agentic runs are not reproducible, so the artifact is
  **not a score** — it's a stream of transcripts + candidate failures. Value is *coverage*. The
  eval remains the regression gate.

## 5. Output: close the loop

Two artifacts per run:

1. **Findings report** — for each probe: the mission, the transcript (every `search`/`ask` call and
   the agent's reasoning), the oracle verdict, and a severity. Human-readable triage.
2. **Candidate gold** — every *confirmed* failure written in the **evaluator's gold schema**
   (`gold_refusal.yaml` / `gold_qa.yaml` shape) as `*.candidate.yaml`, ready for human review and a
   move into `pepper-carrot-eval/data/`. This is the point of the project: discoveries become
   permanent regression tests. **Find once, guard forever.**

## 6. Engineering concerns worth showing

- **Budget governor.** Agent loops are unbounded by default. Cap max turns, max tool calls, and a
  USD ceiling; stop on budget, on "no new failures in N turns," or on mission success. (`governor.py`)
- **`ask` costs real money.** Each `ask` is a real generation against the live app (Post 17's
  honesty note). The governor's USD cap and a `--dry-run`/`--max-tool-calls 1` mode keep it cheap.
- **Determinism boundary is explicit in code.** The agent never computes a verdict; the oracle never
  picks a probe. Keeping those in separate modules (`agent.py` vs `oracle.py`) makes the line
  auditable — the same way the eval keeps dispatch (`run.py`) out of scoring.

## 7. Architecture

```
pepper-carrot-redteam
├─ strategies ......... mission prompts + which oracle each uses    (strategies.py)
├─ agent loop ......... Claude + tool use; PLANS and PICKS probes   (agent.py)   ← agentic core
│     └─ MCP client ... search / ask on the live server            (client.py)
├─ oracle ............. structural (spoiler) + guarded LLM judge    (oracle.py)
├─ governor .......... max turns · tool-call cap · USD ceiling      (governor.py)
└─ outputs ........... findings report + candidate gold (eval schema)  (report.py)
                                   │
                                   ▼  confirmed failures
                       pepper-carrot-eval/data/gold_*.yaml  (after human review)
```

## 8. Build order (tomorrow → on)

1. `config.py`, `client.py` — settle config and prove a tool call against the live server. *(client
   is essentially the eval's wrapper; port it.)*
2. `oracle.py` — port the structural spoiler boundary check + a tiny test. **Do this before the
   agent** so we have a trustworthy verdict to point the agent at.
3. `governor.py` — budget/termination.
4. `agent.py` — the tool-use loop for one strategy (spoiler), wired to the oracle + governor.
5. `report.py` — findings + candidate-gold writer; verify a candidate round-trips into the eval.
6. Second strategy (hallucination) + the LLM-judge oracle.
7. Write Post 19.

## 9. Open questions (decide while building)

- Agent SDK vs. a hand-rolled tool-use loop on the raw Anthropic SDK? (Lean: hand-rolled first, for
  the same "show the engineering" reason the eval avoids a framework — revisit if the loop gets hairy.)
- Multi-turn conversation per probe (reuse a session) vs. fresh session per probe? Spoiler social-
  engineering *wants* multi-turn; reconcile with the eval's "fresh session" rule by scoping reuse to
  a single probe.
- Severity rubric: structural leak = critical; hallucination = medium? Define in `oracle.py`.
