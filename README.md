# pepper-carrot-redteam

> **Status: skeleton.** Design is settled (see [`docs/DESIGN.md`](docs/DESIGN.md) and
> [ADR 0001](docs/decisions/0001-explore-agentically-judge-structurally.md)); the agent loop,
> oracle, and gold hand-off are stubs to be filled in. Tracked as **Post 19** of the
> [Pepper & Carrot AI-powered flipbook](https://bearbearyu1223.github.io/) series.

An **agentic red-teamer** for the deployed [Pepper & Carrot reading companion](https://github.com/bearbearyu1223/pepper-carrot-companion-workshop).

> A deterministic eval can only catch failures you already wrote a test for. An agentic
> red-teamer finds the failures you didn't think of — then hands them to the deterministic
> harness to guard forever.

This is the **discovery** half of evaluation, the complement to the *measurement* half in
[`pepper-carrot-eval`](https://github.com/bearbearyu1223/pepper-carrot-eval). Both are MCP
clients of the same [`pepper-carrot-mcp`](https://github.com/bearbearyu1223/pepper-carrot-mcp)
server and call the same two tools (`search`, `ask`). The difference is who's driving:

| | `pepper-carrot-eval` (Post 18) | `pepper-carrot-redteam` (Post 19) |
|---|---|---|
| Driven by | a **script** (fixed dispatch) | an **agent** (the model picks probes) |
| Goal | **measure** against frozen gold | **discover** unknown failures |
| Output | a reproducible scored report | candidate gold + triaged transcripts |
| Reproducible? | yes (that's the point) | no (coverage, not a number) |

## The one rule it inherits

**Explore agentically; judge structurally.** The agent freely *decides what to try*, but whether
a probe *succeeded* is decided by a checkable oracle wherever possible — never by the same model
that ran the attack. For spoilers that oracle is the exact `(episode, page)` boundary check from
the eval's Instrument 1; only genuinely fuzzy verdicts (e.g. hallucination) fall back to a
guarded LLM judge. See [ADR 0001](docs/decisions/0001-explore-agentically-judge-structurally.md).

## The loop it closes

```
redteam discovers a failure  →  human triages  →  frozen into pepper-carrot-eval gold
        ▲                                                          │
        └──────────────  the eval regression-guards it forever  ◄──┘
```

Find once, guard forever.

## Quick start (once implemented)

```bash
cp .env.example .env        # fill in ANTHROPIC_API_KEY; MCP_SERVER_URL defaults to the live server
uv sync
uv run pepper-carrot-redteam --strategy spoiler --episode 2 --page 3
# → a findings report + any confirmed failures written as *.candidate.yaml
```

## Layout

```
src/pepper_carrot_redteam/
├── config.py       # env-driven settings (server URL, models, budgets, target position)
├── client.py       # MCP client wrapper — search / ask (same tools as the eval)
├── strategies.py   # attack missions (spoiler, hallucination, injection) + their oracles
├── agent.py        # the agentic loop: Claude + tool use decides and adapts the probes
├── oracle.py       # verdicts — structural (spoiler boundary) + guarded LLM judge (fuzzy)
├── governor.py     # budget + termination (max turns / tool calls / USD)
├── report.py       # findings report + candidate-gold writer (eval schema)
└── run.py          # CLI entrypoint
```

## Scope (MVP)

Two strategies — **spoiler** (structural oracle) and **hallucination** (LLM judge) — against one
reader position, behind a budget governor, with confirmed failures written back as candidate gold.
Injection is a stretch goal. See [`docs/DESIGN.md`](docs/DESIGN.md) for the full plan.

## Safety / scope note

This is authorized testing of our own application over public, CC-BY *Pepper & Carrot* content
(© [David Revoy](https://www.peppercarrot.com)). It is a defensive QA tool: it probes our own
deployed companion to harden it, and writes the failures it finds into a regression suite.
