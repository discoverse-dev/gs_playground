from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------ Config: where "state" lives ------------------------
# Try these keys in order. First one found is used.
STATE_KEYS = [
    "state",
    "data.state",
    # 如果你有更深的嵌套（比如 data.xxx.state），把 dot-path 加到这里
]


# ------------------------ Dot-path helpers ------------------------

def get_by_path(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def find_state(row: Dict[str, Any]) -> Optional[List[float]]:
    for k in STATE_KEYS:
        v = get_by_path(row, k) if "." in k else row.get(k, None)
        if isinstance(v, list) and len(v) >= 7:
            return v
    return None


# ------------------------ IO ------------------------

def list_jsonl_files(folder: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(folder, "*.jsonl")))
    return files


def read_jsonl_states(path: str) -> np.ndarray:
    """
    Returns an array of shape (T, 7) for state[0:7] with NaNs if missing.
    """
    out: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e

            if not isinstance(row, dict):
                continue

            st = find_state(row)
            if st is None:
                # keep alignment: append NaNs
                out.append([np.nan] * 7)
                continue

            # cast first 7 dims
            vec = []
            for i in range(7):
                try:
                    vec.append(float(st[i]))
                except Exception:
                    vec.append(np.nan)
            out.append(vec)

    if not out:
        return np.empty((0, 7), dtype=float)
    return np.asarray(out, dtype=float)


# ------------------------ Metrics & summaries ------------------------

@dataclass
class FileSummary:
    file: str
    n_steps: int

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    yaw_min: float
    yaw_max: float

    dx: float
    dy: float
    dz: float
    dyaw: float

    # Selection metrics (based on xyz only, per your “范围正向/反向”描述)
    pos_max_xyz: float   # max positive reach among x,y,z
    neg_min_xyz: float   # most negative reach among x,y,z
    extreme_xyz: float   # max(pos_max_xyz, -neg_min_xyz)


def finite_min(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(a.min()) if a.size else float("nan")


def finite_max(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else float("nan")


def finite_range(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(a.max() - a.min()) if a.size else float("nan")


def summarize_one(path: str) -> FileSummary:
    s = read_jsonl_states(path)
    name = os.path.basename(path)
    n = int(s.shape[0])

    x = s[:, 0] if n else np.array([], dtype=float)
    y = s[:, 1] if n else np.array([], dtype=float)
    z = s[:, 2] if n else np.array([], dtype=float)
    yaw = s[:, 5] if n else np.array([], dtype=float)

    x_min, x_max = finite_min(x), finite_max(x)
    y_min, y_max = finite_min(y), finite_max(y)
    z_min, z_max = finite_min(z), finite_max(z)
    yaw_min, yaw_max = finite_min(yaw), finite_max(yaw)

    dx, dy, dz = finite_range(x), finite_range(y), finite_range(z)
    dyaw = finite_range(yaw)

    pos_max_xyz = np.nanmax([x_max, y_max, z_max]) if np.isfinite([x_max, y_max, z_max]).any() else float("nan")
    neg_min_xyz = np.nanmin([x_min, y_min, z_min]) if np.isfinite([x_min, y_min, z_min]).any() else float("nan")
    extreme_xyz = float("nan")
    if np.isfinite(pos_max_xyz) or np.isfinite(neg_min_xyz):
        a = pos_max_xyz if np.isfinite(pos_max_xyz) else -np.inf
        b = -neg_min_xyz if np.isfinite(neg_min_xyz) else -np.inf
        extreme_xyz = float(max(a, b))

    return FileSummary(
        file=name,
        n_steps=n,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_min=z_min, z_max=z_max,
        yaw_min=yaw_min, yaw_max=yaw_max,
        dx=dx, dy=dy, dz=dz, dyaw=dyaw,
        pos_max_xyz=float(pos_max_xyz),
        neg_min_xyz=float(neg_min_xyz),
        extreme_xyz=float(extreme_xyz),
    )


def summarize_folder(folder: str) -> pd.DataFrame:
    files = list_jsonl_files(folder)
    rows = []
    for fp in files:
        try:
            sm = summarize_one(fp)
            rows.append(sm.__dict__)
        except Exception as e:
            # skip broken files but keep going
            rows.append({"file": os.path.basename(fp), "n_steps": 0, "error": str(e)})
    df = pd.DataFrame(rows)
    return df


def select_top_k_by_extreme_xyz(summary_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    df = summary_df.copy()
    if "extreme_xyz" not in df.columns:
        return df.head(0)

    df = df[pd.to_numeric(df.get("n_steps", 0), errors="coerce").fillna(0) > 0].copy()
    df["extreme_xyz"] = pd.to_numeric(df["extreme_xyz"], errors="coerce")
    df = df.sort_values("extreme_xyz", ascending=False)
    return df.head(top_k).reset_index(drop=True)


def dataset_ranges_from_summaries(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Overall min/max across files using per-file mins/maxes.
    """
    def col_min(c): return pd.to_numeric(df.get(c, np.nan), errors="coerce").min()
    def col_max(c): return pd.to_numeric(df.get(c, np.nan), errors="coerce").max()

    return {
        "x": (float(col_min("x_min")), float(col_max("x_max"))),
        "y": (float(col_min("y_min")), float(col_max("y_max"))),
        "z": (float(col_min("z_min")), float(col_max("z_max"))),
        "yaw": (float(col_min("yaw_min")), float(col_max("yaw_max"))),
    }


# ------------------------ Plotting ------------------------

def _plot_series(ax, t: np.ndarray, y: np.ndarray, label: str, linestyle: str = "-") -> None:
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() >= 2:
        ax.plot(t[m], y[m], linewidth=1, label=label, linestyle=linestyle)


def load_selected_series(folder: str, selected_files: List[str]) -> Tuple[List[Dict[str, np.ndarray]], List[str]]:
    series_list: List[Dict[str, np.ndarray]] = []
    labels: List[str] = []

    for fname in selected_files:
        fp = os.path.join(folder, fname)
        if not os.path.exists(fp):
            continue
        s = read_jsonl_states(fp)
        if s.size == 0:
            continue

        t = np.arange(s.shape[0], dtype=float)
        x, y, z, yaw = s[:, 0], s[:, 1], s[:, 2], s[:, 5]
        series_list.append({"t": t, "x": x, "y": y, "z": z, "yaw": yaw})
        labels.append(os.path.splitext(fname)[0])

    return series_list, labels


def plot_2x2_xyz_yaw_overlay(
    series_list: List[Dict[str, np.ndarray]],
    labels: List[str],
    outpath: str,
    title_prefix: str,
    linestyle: str = "-",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    ax_x, ax_y = axes[0, 0], axes[0, 1]
    ax_z, ax_yaw = axes[1, 0], axes[1, 1]

    for s, lb in zip(series_list, labels):
        _plot_series(ax_x, s["t"], s["x"], lb, linestyle=linestyle)
        _plot_series(ax_y, s["t"], s["y"], lb, linestyle=linestyle)
        _plot_series(ax_z, s["t"], s["z"], lb, linestyle=linestyle)
        _plot_series(ax_yaw, s["t"], s["yaw"], lb, linestyle=linestyle)

    ax_x.set_title(f"{title_prefix} - X vs t")
    ax_y.set_title(f"{title_prefix} - Y vs t")
    ax_z.set_title(f"{title_prefix} - Z vs t")
    ax_yaw.set_title(f"{title_prefix} - Yaw vs t (rad)")

    for ax in [ax_x, ax_y, ax_z, ax_yaw]:
        ax.set_xlabel("t (index)")
    ax_x.set_ylabel("X")
    ax_y.set_ylabel("Y")
    ax_z.set_ylabel("Z")
    ax_yaw.set_ylabel("Yaw (rad)")

    # legend only on first subplot to reduce clutter
    ax_x.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_compare_2x2_xyz_yaw(
    real_series: List[Dict[str, np.ndarray]],
    real_labels: List[str],
    sim_series: List[Dict[str, np.ndarray]],
    sim_labels: List[str],
    outpath: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    ax_x, ax_y = axes[0, 0], axes[0, 1]
    ax_z, ax_yaw = axes[1, 0], axes[1, 1]

    # Real: solid
    for s, lb in zip(real_series, real_labels):
        _plot_series(ax_x, s["t"], s["x"], f"real:{lb}", linestyle="-")
        _plot_series(ax_y, s["t"], s["y"], f"real:{lb}", linestyle="-")
        _plot_series(ax_z, s["t"], s["z"], f"real:{lb}", linestyle="-")
        _plot_series(ax_yaw, s["t"], s["yaw"], f"real:{lb}", linestyle="-")

    # Sim: dashed
    for s, lb in zip(sim_series, sim_labels):
        _plot_series(ax_x, s["t"], s["x"], f"sim:{lb}", linestyle="--")
        _plot_series(ax_y, s["t"], s["y"], f"sim:{lb}", linestyle="--")
        _plot_series(ax_z, s["t"], s["z"], f"sim:{lb}", linestyle="--")
        _plot_series(ax_yaw, s["t"], s["yaw"], f"sim:{lb}", linestyle="--")

    ax_x.set_title("Compare - X vs t")
    ax_y.set_title("Compare - Y vs t")
    ax_z.set_title("Compare - Z vs t")
    ax_yaw.set_title("Compare - Yaw vs t (rad)")

    for ax in [ax_x, ax_y, ax_z, ax_yaw]:
        ax.set_xlabel("t (index)")
    ax_x.set_ylabel("X")
    ax_y.set_ylabel("Y")
    ax_z.set_ylabel("Z")
    ax_yaw.set_ylabel("Yaw (rad)")

    ax_x.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ------------------------ Main runner ------------------------

def run(real_dir: str, sim_dir: str, output_dir: str, top_k: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Summaries
    real_sum = summarize_folder(real_dir)
    sim_sum = summarize_folder(sim_dir)

    real_sum.to_csv(os.path.join(output_dir, "summary_real.csv"), index=False, encoding="utf-8")
    sim_sum.to_csv(os.path.join(output_dir, "summary_sim.csv"), index=False, encoding="utf-8")

    # Selection: real by extreme_xyz (covers positive/negative)
    real_sel = select_top_k_by_extreme_xyz(real_sum, top_k)
    # Sim selection: use same selection rule to keep plots comparable
    sim_sel = select_top_k_by_extreme_xyz(sim_sum, top_k)

    real_sel.to_csv(os.path.join(output_dir, "selected_real.csv"), index=False, encoding="utf-8")
    sim_sel.to_csv(os.path.join(output_dir, "selected_sim.csv"), index=False, encoding="utf-8")

    # Load series
    real_files = real_sel["file"].tolist() if "file" in real_sel.columns else []
    sim_files = sim_sel["file"].tolist() if "file" in sim_sel.columns else []

    real_series, real_labels = load_selected_series(real_dir, real_files)
    sim_series, sim_labels = load_selected_series(sim_dir, sim_files)

    # Figures (2x2 each)
    if sim_series:
        plot_2x2_xyz_yaw_overlay(
            sim_series, sim_labels,
            os.path.join(output_dir, "fig_sim_2x2_xyz_yaw.png"),
            title_prefix="SIM (selected)",
            linestyle="-",
        )

    if real_series:
        plot_2x2_xyz_yaw_overlay(
            real_series, real_labels,
            os.path.join(output_dir, "fig_real_2x2_xyz_yaw.png"),
            title_prefix="REAL (selected)",
            linestyle="-",
        )

    if real_series and sim_series:
        plot_compare_2x2_xyz_yaw(
            real_series, real_labels, sim_series, sim_labels,
            os.path.join(output_dir, "fig_compare_2x2_xyz_yaw.png"),
        )

    # Range report (overall + selected)
    real_overall = dataset_ranges_from_summaries(real_sum)
    sim_overall = dataset_ranges_from_summaries(sim_sum)

    real_selected_ranges = dataset_ranges_from_summaries(real_sel) if len(real_sel) else {}
    sim_selected_ranges = dataset_ranges_from_summaries(sim_sel) if len(sim_sel) else {}

    report_path = os.path.join(output_dir, "ranges_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        def w(title: str):
            f.write(title + "\n")
            f.write("-" * len(title) + "\n")

        def dump_ranges(tag: str, rr: Dict[str, Tuple[float, float]]):
            f.write(f"{tag}\n")
            for k in ["x", "y", "z", "yaw"]:
                if k in rr:
                    mn, mx = rr[k]
                    f.write(f"  {k}: [{mn:.6f}, {mx:.6f}]  range={mx - mn:.6f}\n")
            f.write("\n")

        w("OVERALL RANGES (across ALL files)")
        dump_ranges("REAL overall", real_overall)
        dump_ranges("SIM  overall", sim_overall)

        w(f"SELECTED RANGES (top_k={top_k} by extreme_xyz)")
        dump_ranges("REAL selected", real_selected_ranges)
        dump_ranges("SIM  selected", sim_selected_ranges)

        w("SELECTION CRITERION")
        f.write(
            "Per-file metrics computed on xyz only:\n"
            "  pos_max_xyz = max(x_max, y_max, z_max)\n"
            "  neg_min_xyz = min(x_min, y_min, z_min)\n"
            "  extreme_xyz = max(pos_max_xyz, -neg_min_xyz)\n"
            "Selected top_k by extreme_xyz descending.\n"
        )

    print(f"[OK] Wrote outputs to: {output_dir}")
    print(f" - {os.path.join(output_dir, 'summary_real.csv')}")
    print(f" - {os.path.join(output_dir, 'summary_sim.csv')}")
    print(f" - {os.path.join(output_dir, 'selected_real.csv')}")
    print(f" - {os.path.join(output_dir, 'selected_sim.csv')}")
    print(f" - {os.path.join(output_dir, 'ranges_report.txt')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True, help="Folder for real robot jsonl files (e.g., 2026-01-20.jsonl)")
    ap.add_argument("--sim_dir", required=True, help="Folder for sim jsonl files (e.g., episodexxxx*.jsonl)")
    ap.add_argument("--output_dir", required=True, help="Output folder")
    ap.add_argument("--top_k", type=int, default=10, help="Number of selected files for plotting and range report")
    args = ap.parse_args()

    top_k = max(1, int(args.top_k))
    run(args.real_dir, args.sim_dir, args.output_dir, top_k)


if __name__ == "__main__":
    main()
