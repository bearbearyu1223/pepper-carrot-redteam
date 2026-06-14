# pepper-carrot-redteam

> **Status: implemented** (both phases), green on `ruff` · `mypy --strict` · `pytest`. Four
> strategies — **spoiler**, **hallucination**, **injection**, **blindspot** — drive the multi-turn
> agent loop, behind the budget governor, with confirmed failures written back as candidate gold and
> a per-probe JSONL forensic trace — plus a metered Break-Rate experiment harness for statistical
> robustness measurement. Design lives in the build plan
> ([`docs/DESIGN.md`](docs/DESIGN.md)) and two decision records:
> [ADR 0001](docs/decisions/0001-explore-agentically-judge-structurally.md) (explore agentically,
> judge structurally) and
> [ADR 0002](docs/decisions/0002-multi-turn-social-engineering-and-guarded-spoiler-judge.md)
> (multi-turn social engineering + a guarded spoiler-leak judge). Tracked as **Post 19** of the
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

## Quick start

```bash
cp .env.example .env        # fill in ANTHROPIC_API_KEY; MCP_SERVER_URL defaults to the live server
uv sync

uv run pepper-carrot-redteam --strategy spoiler --episode 2 --page 3
uv run pepper-carrot-redteam --strategy hallucination
uv run pepper-carrot-redteam --strategy spoiler --dry-run     # 1 probe, no gold written (cheap smoke)
uv run pepper-carrot-redteam --strategy spoiler --multi-turn  # force every ask to continue the session
```

Each run writes a Markdown findings report to `findings/<run-id>.md` and, for any **confirmed**
failures, candidate gold to `EVAL_GOLD_DIR` as `redteam-<oracle>-<run-id>.candidate.yaml` (skipped
under `--dry-run`). Use `-v` for per-probe progress (tool · intent · verdict · live budget) and
`-vv` to log every `search`/`ask`/judge call with latencies and session continuity — the debugging
view. `--max-tool-calls N` overrides the budget cap for a single run. `--multi-turn` forces every
`ask` probe to continue the same server session (overriding the agent's per-probe choice), so the
multi-turn pressure path is exercised deterministically instead of only when the agent opts in — it
affects the ask-based strategies (spoiler, hallucination, injection) and is a no-op for `blindspot`.

## Layout

```
src/pepper_carrot_redteam/
├── config.py       # env-driven settings (server URL, models, budgets + stall, target position)
├── client.py       # MCP client wrapper — search / ask (same tools as the eval)
├── strategies.py   # attack missions: spoiler · hallucination · injection · blindspot
├── agent.py        # the agentic loop: Claude decides/adapts probes (multi-turn, obfuscate, language)
├── oracle.py       # verdicts — structural (spoiler boundary, blindspot) + guarded LLM judges (fuzzy)
├── governor.py     # budget + termination (max turns / tool calls / USD / stall patience)
├── tracing.py      # per-probe JSONL forensic record
├── report.py       # findings report + candidate-gold writer (eval schema)
├── experiment.py   # metered Break-Rate experiment: strategies × positions × reps (statistical eval)
├── analysis.py     # re-analyze saved runs.jsonl: Break Rate + CIs, A/B ablation w/ significance test
└── run.py          # CLI entrypoint
```

## Strategies

| strategy | tools | oracle | confirmed failure → eval gold |
|---|---|---|---|
| **spoiler** | `search` + multi-turn `ask` | structural boundary check ∥ guarded spoiler-leak judge | `gold_refusal` kind:spoiler |
| **hallucination** | multi-turn `ask` (+ paired wiki `search` for grounding) | guarded groundedness judge | `gold_refusal` kind:unanswerable |
| **injection** | multi-turn `ask` (+ `obfuscate`: base64/rot13/leetspeak/homoglyph/zero_width; + `language`: cross-lingual) | structural (boundary-widening) ∥ guarded out-of-domain judge | `gold_refusal` kind:spoiler / out_of_domain |
| **blindspot** | `probe_retrieval` (returns rank/score feedback to hill-climb) | semi-structural paraphrase-divergence (no model) | `gold_retrieval` (query + dropped chunk) |

All against one reader position, behind a budget governor, with confirmed failures written back as
candidate gold. See [`docs/DESIGN.md`](docs/DESIGN.md) §3–§5 for the full design and ADR 0001/0002
for the load-bearing decisions.

## Measuring robustness: the experiment harness

A single agentic run is *discovery* — coverage, not a number. Repeat it many times and aggregate,
though, and you get a statistical robustness measurement. [`experiment.py`](src/pepper_carrot_redteam/experiment.py)
runs a grid of **strategies × reader-positions × replicates** and reports the **Break Rate** (the
fraction of runs that surface a confirmed failure) per strategy, with **Wilson 95% confidence
intervals** — plus the *real* dollar cost (metered, not the governor's notional budget).

```bash
# $0 pipeline check — fakes the model + MCP, no key/network:
uv run python -m pepper_carrot_redteam.experiment --mock --reps 2

# a moderate grid: 4 strategies × 3 positions × 10 reps = 120 runs:
uv run python -m pepper_carrot_redteam.experiment --positions "3:2,9:5,11:4" --reps 10
```

It prints a per-strategy Break-Rate table, a **strategy × position matrix** (the weak-spot heatmap),
and a cost breakdown; per-run rows land in `experiments/<exp-id>/runs.jsonl`. Add `-v` for per-turn
logs and `--trace` for a full JSONL trace per run. The governor caps (`MAX_TURNS` / `MAX_TOOL_CALLS`
/ `MAX_USD` / `STALL_PATIENCE`) and `TARGET_EPISODE`/`PAGE` come from `.env` unless a flag overrides.

[`analysis.py`](src/pepper_carrot_redteam/analysis.py) re-analyzes saved `runs.jsonl` (no re-spending):
combine several experiments into one report, or run an **A/B ablation** with a two-proportion
significance test — e.g. "does `--multi-turn` raise the spoiler Break Rate, and is the gap real?"

```bash
uv run python -m pepper_carrot_redteam.analysis experiments/exp-2026*/runs.jsonl
uv run python -m pepper_carrot_redteam.analysis experiments/baseline --vs experiments/multiturn
```

**Ablation example — does forcing multi-turn make it leak more?** The experiment honors `--multi-turn`
(forces every `ask` to continue one session, across the whole grid). Run the grid twice — baseline vs
forced multi-turn — then ablate with a two-proportion z-test:

```bash
# A: baseline (the agent decides whether to continue)
uv run python -m pepper_carrot_redteam.experiment --positions "3:2,9:5" --reps 15
mv experiments/exp-* experiments/baseline

# B: forced multi-turn (every ask continues the same session)
uv run python -m pepper_carrot_redteam.experiment --positions "3:2,9:5" --reps 15 --multi-turn
mv experiments/exp-* experiments/multiturn

# is the multi-turn lift real? → per-strategy ΔBreak Rate + significance
uv run python -m pepper_carrot_redteam.analysis experiments/baseline --vs experiments/multiturn
```

(`--multi-turn` only affects the ask-based strategies — spoiler, hallucination, injection; it's a
no-op for `blindspot`.)

**Cost is two-sided, on one account.** Your `ANTHROPIC_API_KEY` pays for both the client-side calls
(the agent + the judges — metered exactly from the SDK `usage`) *and* the server-side companion
generation behind each `ask` (Haiku, estimated by call-count × `--ask-cost`). The harness reports the
split and projects a full-grid cost from the measured per-run cost.

> The metric is **Break Rate** (report its complement, **Hold Rate** = 1 − Break Rate). Trust the
> structural-oracle Break Rates most (no model in the verdict); the guarded-judge ones are softer, so
> a small human judge-calibration sample is worth doing before reading too much into them. This is
> still *measurement built on discovery* — a Break Rate with CIs, distinct from the eval's
> deterministic score.

## Safety / scope note

This is authorized testing of our own application over public, CC-BY *Pepper & Carrot* content
(© [David Revoy](https://www.peppercarrot.com)). It is a defensive QA tool: it probes our own
deployed companion to harden it, and writes the failures it finds into a regression suite.
