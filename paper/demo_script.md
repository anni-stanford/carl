# CARL — 3-minute demo script

| Time | Beat |
|---|---|
| 0:00–0:25 | **Problem.** Coding agents have skills/configs but nobody systematically tunes them. Show a generic `CLAUDE.md` and a generic `.cursor/rules`. |
| 0:25–1:00 | **CARL.** RL formulation. Four-technique stack (RLVR + GRPO + DPO + bandits). The async loop. |
| 1:00–2:00 | **Live demo.** Streamlit dashboard. Reward curves climb for **both** Claude Code and Cursor as CARL evolves their respective artifacts on a real repo (FastAPI). Show 2 promoted diffs per adapter with reward deltas + CIs. |
| 2:00–2:30 | **Main results.** SWE-Bench Verified subset, both adapters, with confidence intervals. Cross-IDE transfer result (if E7 ready) or roadmap if deferred. |
| 2:30–3:00 | **Limitations honestly.** Open-source link. Cross-IDE pitch: "CARL is the optimization layer above whichever coding agent you use." Citation. |
