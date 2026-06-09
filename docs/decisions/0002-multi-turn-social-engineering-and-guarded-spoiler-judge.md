# ADR 0002 — Multi-turn social engineering, and a guarded spoiler-leak judge alongside the structural oracle

- **Status:** Accepted
- **Date:** 2026-06-08
- **Context:** scoping the agent loop for `pepper-carrot-redteam` (Post 19). Amends — does not
  overturn — [ADR 0001](0001-explore-agentically-judge-structurally.md).

## Context

ADR 0001 settled the load-bearing rule: the agent decides actions, an oracle decides verdicts, and
spoiler verdicts are **structural** (a checkable predicate on retrieved `(episode, page)` keys), with
a guarded LLM judge reserved for "genuinely fuzzy" modes like hallucination. The MVP sketch had each
probe be a single `search` call: the agent's only lever was query phrasing.

Two gaps surfaced while scoping the loop:

1. **The compelling spoiler artifact is a *conversation*, not a query.** The whole series stakes
   itself on spoiler-safety, and the strongest demonstration is an agent that *socially engineers*
   the companion — roleplay, rapport, incremental coercion across turns — and is still held at the
   boundary. A single stateless `search` can't exercise that. The server supports it: `ask` returns
   a `session_id`, and the chat pipeline replays prior turns into the prompt, so reusing that id
   gives the companion real conversational memory.
2. **A prose leak can live where the structural check can't see it.** The structural oracle inspects
   *retrieval* positions. But a companion could narrate the ending in prose even when page-mode
   retrieval is clean (parametric knowledge, an over-eager summary). The structural check, by
   construction, cannot catch a leak that never appears as a retrieved past-boundary chunk.

## Decision

1. **Probes may be multi-turn.** The agent can reuse `ask`'s `session_id` to build conversational
   pressure within a single probe-conversation. The **harness owns the session id**; the agent gets
   only a `continue_session` flag (continue the current conversation, or start a clean angle). The
   reader position stays **pinned server-side on every turn** — the agent never controls it, so a
   leak remains a real finding, not a self-granted permission.

2. **Spoiler gets a dual oracle.**
   - **Structural stays primary.** The `(episode, page)` boundary check on `search` results
     (ADR 0001's ported `_past_boundary`) remains the high-confidence verdict — `basis="structural"`.
   - **A new guarded spoiler-leak judge** (`judge_spoiler_leak`) runs on `ask` answers to catch
     *prose* leaks the structural check can't see, reusing ADR 0001's judge guards (cross-model,
     anchored rubric, forced structured output, temperature 0). Flagged lower-confidence —
     `basis="judged"`.
   - **Precedence:** a turn `failed = structural OR judged`; `basis` prefers `"structural"` when the
     search check fires.

## Why this is consistent with ADR 0001

- **Structural is still preferred where the failure mode is structural.** We did not replace the
  structural check; we added a judge only for the *generative* failure mode — a leak that lives in
  prose, not in retrieved positions. That is exactly the "genuinely fuzzy" carve-out ADR 0001
  already grants hallucination.
- **The attacker still does not judge itself.** The spoiler-leak judge is a separate, cross-model
  call (`judge_model`), never the agent model. The determinism boundary (agent decides, oracle
  judges; separate modules) is unchanged.
- **The reproducibility caveat is unchanged.** Multi-turn runs are *less* reproducible, not more;
  the artifact remains coverage (transcripts + candidate gold), and the deterministic eval remains
  the regression gate.

## Consequences

- **More compelling artifact:** social-engineered spoiler attempts, with a structural proof the
  boundary held (and a judged signal when prose overreaches).
- **More cost:** multi-turn means more `ask` calls. The budget governor (max turns / tool calls /
  USD) is doing more work, not less — its caps and `--dry-run` matter more, not less.
- **Intentional session bleed.** The eval's "fresh session per call" rule exists to keep scoring
  clean; here, session continuity *is* the attack. Scope the bleed to a single probe-conversation
  (a `continue_session=false` starts fresh), so a leak is still attributable to one conversational
  arc.
- **Trace by session.** Records carry the `session_id` so a whole social-engineering conversation is
  greppable as one arc (see the tracer in DESIGN §11).

## Alternatives considered

- **Structural-only spoiler (keep ADR 0001's MVP sketch).** Rejected: can't catch prose leaks, and —
  more importantly — can't exercise the social-engineering demo that is the point of an *agentic*
  red-teamer. Structure alone under-sells what an adaptive attacker can do.
- **Single-turn probes.** Rejected: forecloses the conversational coercion that makes the spoiler
  finding credible and interesting.
- **Let the judge be the agent model.** Rejected for the same reason as ADR 0001 — an attacker is
  not a trustworthy judge of its own success. Cross-model only.
