"""Analyze Webots simulation telemetry CSV logs and generate report-ready charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_latest_log(logs_dir: Path) -> Path:
    csvs = sorted(logs_dir.glob("run_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No run_*.csv logs found in {logs_dir}")
    return csvs[-1]


def analyze(csv_path: Path, out_dir: Path | None = None, lane_half_width_m: float = 1.0) -> Path:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Telemetry file is empty: {csv_path}")

    out_dir = out_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    lane_df = df[df["fsm_state"] == "LANE_FOLLOWING"]
    if lane_df.empty:
        lane_df = df

    abs_cte = lane_df["cte_m"].abs()
    mean_cte = float(abs_cte.mean())
    max_cte = float(abs_cte.max())
    in_lane_pct = float((abs_cte <= lane_half_width_m).mean() * 100.0)
    avg_fps = float(df["fps"].mean())
    avg_yolo = float(df["yolo_ms"].mean())
    avg_lane = float(df["lane_ms"].mean())

    print("\n================ SIMULATION RUN KPI ================")
    print(f"Log file:           {csv_path.name}")
    print(f"Duration:           {df['sim_time_s'].max() - df['sim_time_s'].min():.1f} s ({len(df)} frames)")
    print(f"Mean |CTE|:         {mean_cte:.3f} m")
    print(f"Max  |CTE|:         {max_cte:.3f} m")
    print(f"In-Lane Rate:       {in_lane_pct:.1f}% (|CTE| <= {lane_half_width_m:.1f} m)")
    print(f"Average FPS:        {avg_fps:.1f} FPS")
    print(f"Avg Inference:      YOLO {avg_yolo:.2f} ms | Lane CV {avg_lane:.2f} ms")
    print("Signs observed:    ", sorted(s for s in df["detected_sign"].dropna().unique() if s != "NONE"))
    print("====================================================\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Autonomous Robot Telemetry Report — {csv_path.stem}", fontsize=13, fontweight="bold")

    # 1. Cross-Track Error over time
    ax = axes[0, 0]
    ax.plot(df["sim_time_s"], df["cte_m"], label="CTE (m)", color="#1f77b4", linewidth=1.2)
    ax.axhline(lane_half_width_m, color="red", linestyle="--", alpha=0.6, label="±Lane threshold")
    ax.axhline(-lane_half_width_m, color="red", linestyle="--", alpha=0.6)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax.set_title(f"Cross-Track Error (Mean |CTE| = {mean_cte:.3f} m, In-Lane = {in_lane_pct:.1f}%)")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("CTE (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Speed vs Target Speed
    ax = axes[0, 1]
    ax.plot(df["sim_time_s"], df["speed_kmh"], label="Actual Speed", color="#2ca02c", linewidth=1.3)
    ax.plot(
        df["sim_time_s"],
        df["target_speed_kmh"],
        label="Target Speed",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.2,
    )
    ax.set_title("Speed Control & Traffic Sign Response")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. 2D Trajectory (GPS)
    ax = axes[1, 0]
    sc = ax.scatter(df["pos_x"], df["pos_y"], c=df["speed_kmh"], cmap="viridis", s=5)
    fig.colorbar(sc, ax=ax, label="Speed (km/h)")
    ax.set_title("Robot 2D Trajectory (GPS)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    # 4. Real-time Latency & FPS
    ax = axes[1, 1]
    ax.plot(df["sim_time_s"], df["yolo_ms"], label=f"YOLO ({avg_yolo:.1f} ms)", color="#d62728", linewidth=1.1)
    ax.plot(df["sim_time_s"], df["lane_ms"], label=f"Lane CV ({avg_lane:.1f} ms)", color="#9467bd", linewidth=1.1)
    ax.set_title(f"Processing Latency (Avg FPS = {avg_fps:.1f})")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Latency (ms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = out_dir / f"{csv_path.stem}_report.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    print(f"Saved telemetry chart to: {chart_path}")
    return chart_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze telemetry CSV log from Webots run.")
    parser.add_argument("--csv", type=Path, default=None, help="Path to CSV log (defaults to latest in logs/).")
    args = parser.parse_args()

    csv_path = args.csv if args.csv is not None else find_latest_log(PROJECT_ROOT / "logs")
    analyze(csv_path)


if __name__ == "__main__":
    main()
