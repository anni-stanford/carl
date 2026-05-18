# CS 153 Project Milestone

Anni Zimina, May 17 2026.

## Q1. Project title

CARL: a reinforcement learning framework that improves Claude Code's `CLAUDE.md`, `.claude/skills/`, sub-agents, hooks, and `settings.json` by running episodes against real repositories and scoring the outcome with the project's CI pipeline.

## Q2. Project track

Research.

## Q3. Progress

The repo is at github.com/anni-stanford/carl. Most of the algorithmic stack is in and working: an RLVR-style verifier that parses real pytest reports, coverage XML, ruff JSON, and mypy output and renormalizes its weights when a signal is missing; an LLM-judge with position-flip, family rotation across Claude Opus, GPT-5.5, and Claude Sonnet, and rubric-shuffle for bias control; six-pattern adversarial reward-hacking probes; a GRPO-style group-relative advantage scorer; a Thompson-sampling contextual bandit; a Pydantic-validated structured-output diagnosis agent; a locality-bounded mutation proposer; a DPO-style preference ranker; an `apply_diff` function that actually edits the on-disk `CLAUDE.md` and `.claude/skills/` files when a candidate clears the gate; and the paired-bootstrap promotion gate itself, which uses scipy's BCa bootstrap with 10,000 resamples and only promotes when the 95 % CI lower bound on the per-task reward lift exceeds zero. There is a single command, `carl auto`, that runs the whole pipeline end-to-end: it benchmarks the user's current `CLAUDE.md` on a small task set (n=10 paired tasks by default), runs training episodes during which CARL evolves the `CLAUDE.md` and skills, benchmarks again on the same tasks, runs the gate, and writes a markdown report with the lift number, CI bounds, p-value, decomposition, and the diff history of what changed. 83 unit and integration tests pass under Python 3.11 and 3.12, `ruff` and `mypy --strict` are clean across 43 source files, and CI is green on the latest commit (`52de879`, although there's also a more recent fix to make `python -m carl` work without `PATH` munging). The thing I have not done yet is the real-data run: building the `carl/episode-claude:latest` Docker image, wiring it to my Anthropic API key, and producing real numbers to replace the placeholders in the results table.

## Q4. Future implementation

Between now and May 29 I am planning to do exactly the real-data run that the milestone is missing.

May 18 to 22 is for getting one real episode to run end-to-end. The Dockerfile is committed at `docker/Dockerfile.episode.claude` but I have not actually built and tested it on my laptop yet. I'll install the Claude Code CLI in the image (the npm package is `@anthropic-ai/claude-code`), confirm the in-container wrapper produces the expected pytest JSON / coverage XML / ruff JSON / mypy output files in the artifact mount, and then run a single episode on a small Python repo (probably FastAPI or httpx) to confirm the trajectory, the verifier, and the buffer all see what they expect.

May 23 to 25 is the main experiment. I'll register the FastAPI repo, queue 30 paired tasks (mostly drawn from real closed GitHub issues so they have known-good solutions for verifier comparison), run the stock-Claude-Code baseline on all 30, then run 50 training episodes during which CARL proposes and gates `CLAUDE.md` and skill changes, then re-run all 30 against the evolved policy. The output is the paired-rewards table that goes into `experiments/ab_compare.py` and produces the headline E1 row with mean lift, BCa CI, and p-value.

May 26 to 27 is for two ablations: I'll re-run the same 30 paired tasks with the diagnosis agent disabled (random artifact selection) and with the reward-hacking penalty disabled (no `r_hack` term). Each ablation gets its own paired-bootstrap CI against full-CARL, which gives a per-technique contribution number for the paper.

May 28 is for paper writing — replacing every placeholder in `paper/draft.md` with real numbers from the buffer, regenerating the dashboard screenshots, and writing the limitations section against what the experiments actually showed (rather than what I expect them to show). May 29 is for recording the 3-minute demo video, tagging `v0.1.0`, and publishing `carl-loop` on PyPI so `pip install carl-loop` works without git.

I am deliberately not attempting cross-language transfer, full SWE-Bench Verified, or the DPO classifier (replacing the v1 structured-prompt ranker) within this window. Those experiments need more compute time and would risk under-powering the headline result. The README and `paper/draft.md` flag them as v0.2 work.

## Q5. Github link

https://github.com/anni-stanford/carl

The README's quickstart is one shell line: a `curl ... | bash` that auto-installs and runs `carl auto` against the current directory, falling back to `--dry-run` if the user does not have Docker or an Anthropic API key. There is also a manual three-step path for users who want to read the installer first. The repo's `paper/milestone_may17.md` mirrors this document.

## Q6. Compute

I do not need a GPU cluster. CARL operates on text artifacts, not on model weights. The estimated cost across the May 17 to 29 window is around $3-5K, all comfortably inside the available student credits.

The bulk of the cost is Anthropic API for Claude Opus 4.7, used as the mutator, the LLM judge, and the diagnosis agent. I estimate 3,000 to 5,000 calls at roughly $0.40 average, so $1.5-2K. The Cloudflare for Startups program gives me $50K of Workers AI credits and I expect to use $1-2K of that for the open-weight family-rotation in the LLM judge (this is the bias-control mechanism that lets me defend against same-family judge-mutator collusion). DigitalOcean's $250 student credit covers the SQLite replay buffer host and the Streamlit dashboard for around $60 a month during the window. If I had to allocate one extra ask, it would be additional Anthropic API budget in case the real run needs more than the planned 80 episodes; the rest is sufficient.

---

*AI usage disclosure (CS 153 AI Policy):* I used large-language-model coding assistants to help draft initial scaffolding code, documentation, and test fixtures. The design decisions, the RL stack implementation, the statistical methodology, the experimental analysis, the paper writing, and the final code review are mine.
