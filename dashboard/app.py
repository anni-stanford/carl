"""CARL Streamlit dashboard.

Reads the SQLite replay buffer and renders four views: live loop status,
reward curve over time, gate-decision (promotion) history, and per-episode
replay. Run with::

    streamlit run dashboard/app.py -- --buffer carl_run/buffer.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=Path, default=Path("carl_run/buffer.sqlite"))
    return parser.parse_args(sys.argv[1:])


def _load(buffer: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not buffer.is_file():
        return pd.DataFrame(), pd.DataFrame()
    with sqlite3.connect(str(buffer)) as conn:
        traj = pd.read_sql_query(
            "SELECT created_at, adapter_name, task_id, policy_version, "
            "r_total, r_verifier, r_judge, r_hack, exit_code, duration_s "
            "FROM trajectories ORDER BY created_at",
            conn,
        )
        gates = pd.read_sql_query(
            "SELECT created_at, candidate_version, baseline_version, promote, "
            "mean_lift, ci_low, ci_high, p_value, n_tasks, reason "
            "FROM gate_decisions ORDER BY created_at",
            conn,
        )
    return traj, gates


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="CARL Dashboard", layout="wide")
    st.title("CARL — Continuous Agent Reinforcement Loop")
    st.caption(f"buffer: `{args.buffer}`")

    traj, gates = _load(args.buffer)
    if traj.empty:
        st.warning("No trajectories in buffer yet. Run an experiment first.")
        return

    tab_overview, tab_reward, tab_promotions, tab_replay = st.tabs(
        ["Overview", "Reward curve", "Promotions", "Episode replay"]
    )

    with tab_overview:
        col1, col2, col3 = st.columns(3)
        col1.metric("Trajectories", f"{len(traj):,}")
        col2.metric("Promotions", int(gates["promote"].sum()) if not gates.empty else 0)
        col3.metric("Mean reward", f"{traj['r_total'].mean():.4f}")
        st.subheader("Latest 20 trajectories")
        st.dataframe(traj.tail(20), use_container_width=True)

    with tab_reward:
        st.subheader("Reward over time, per policy version")
        traj["created_at"] = pd.to_datetime(traj["created_at"], errors="coerce")
        pivot = traj.pivot_table(
            index="created_at", columns="policy_version", values="r_total"
        )
        st.line_chart(pivot)
        st.subheader("Reward decomposition (mean per policy version)")
        decomp = traj.groupby("policy_version")[["r_verifier", "r_judge", "r_hack"]].mean()
        st.bar_chart(decomp)

    with tab_promotions:
        st.subheader("Gate decisions")
        if gates.empty:
            st.info("No gate decisions yet.")
        else:
            st.dataframe(gates, use_container_width=True)

    with tab_replay:
        st.subheader("Episode replay")
        if traj.empty:
            st.info("No episodes available.")
        else:
            ids = traj["task_id"].unique().tolist()
            tid = st.selectbox("Task ID", ids)
            view = traj[traj["task_id"] == tid]
            st.dataframe(view, use_container_width=True)


if __name__ == "__main__":  # pragma: no cover
    main()
