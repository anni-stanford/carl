# Reproducing CARL results

```bash
git clone https://github.com/anni-stanford/carl
cd carl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, optional CURSOR_API_KEY
docker compose -f docker/compose.yml up -d  # Postgres + dashboard

# Smoke test (no real episodes, just the loop wiring + tests)
pytest -q

# Reproduce E1 (main result) — Claude Code adapter on FastAPI
bash scripts/run_all.sh --experiment e1 --adapter claude_code --repo fastapi
```

Each experiment writes its results to `experiments/results/<exp>/` and the dashboard reads from there. The paper draft (`paper/draft.md`) has table cells linked to those files.
