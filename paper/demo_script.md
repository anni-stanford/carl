# CARL — 3-minute demo script

| Time | Beat |
|---|---|
| 0:00–0:25 | **Problem.** Coding agents have skills/configs but nobody systematically tunes them. Show a stock generic `CLAUDE.md` on a real repo. |
| 0:25–1:00 | **CARL.** RL formulation. Four-technique stack (RLVR + GRPO + DPO + bandits). The async loop. |
| 1:00–2:00 | **Live demo.** Streamlit dashboard. Reward curve climbs as CARL evolves the `CLAUDE.md` and `.claude/skills/` of a real repo (FastAPI). Show 2 promoted diffs with reward deltas + CIs. |
| 2:00–2:30 | **Main results.** SWE-Bench Verified subset, paired-bootstrap CI, _p_-value. |
| 2:30–3:00 | **Limitations honestly.** Open-source link. Pitch: "CARL is the optimization layer above your CLAUDE.md." Citation. |
