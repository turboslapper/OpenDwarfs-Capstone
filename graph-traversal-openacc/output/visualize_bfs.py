import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load + basic cleanup (OpenACC BFS CSV)
# -----------------------------
CSV = "bfs_profile.csv"
df = pd.read_csv(CSV)
df.columns = df.columns.str.strip()

threads = [1, 2, 4, 8, 16, 32]
time_cols  = [f"ms_t{t}" for t in threads]
mteps_cols = [f"mteps_t{t}" for t in threads]

# Ensure numeric
for c in time_cols + ["num_nodes", "num_edges", "avg_degree"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# If mteps columns exist, parse them too
for c in mteps_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows missing required fields
required = ["graph", "num_nodes", "num_edges"] + time_cols
df = df.dropna(subset=required).copy()

# -----------------------------
# Helpers: recompute metrics from TIMES (source of truth)
# -----------------------------
def times_for_row(row):
    return np.array([row[f"ms_t{t}"] for t in threads], dtype=float)

def speedups_from_times(times_ms):
    # speedup(t) = T1 / Tt (includes t=1 => 1)
    return times_ms[0] / times_ms

def efficiency_from_speedups(speedups):
    return (speedups / np.array(threads)) * 100.0

def mteps_from_times(num_edges, times_ms):
    # MTEPS = (edges / seconds) / 1e6 ; seconds = ms/1000
    return (num_edges / (times_ms / 1000.0)) / 1e6

# Recompute speedup@32 for sorting
df["speedup32_recalc"] = df.apply(
    lambda r: speedups_from_times(times_for_row(r))[threads.index(32)],
    axis=1
)

# Top 3 by recomputed speedup@32
df_top = df.sort_values("speedup32_recalc", ascending=False).head(3).reset_index(drop=True)

# -----------------------------
# Plot: 2x2 summary for top 3
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("BFS OpenACC Performance (Top 3 by 32-thread Speedup)", fontsize=16, fontweight="bold")

def label_for(row):
    return f"{row['graph']} (speedup@32 = {row['speedup32_recalc']:.2f}×)"

# 1) Execution time vs threads
ax1 = axes[0, 0]
for _, row in df_top.iterrows():
    times = times_for_row(row)
    ax1.plot(threads, times, marker="o", linewidth=2, markersize=7, label=label_for(row))
ax1.set_title("Execution Time vs Thread Count", fontsize=13, fontweight="bold")
ax1.set_xlabel("Number of Threads")
ax1.set_ylabel("Execution Time (ms)")
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xticks(threads)
ax1.set_xticklabels(threads)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2) Speedup vs threads (include ideal)
ax2 = axes[0, 1]
for _, row in df_top.iterrows():
    times = times_for_row(row)
    speedups = speedups_from_times(times)
    ax2.plot(threads[1:], speedups[1:], marker="s", linewidth=2, markersize=7, label=label_for(row))
ax2.plot(threads[1:], threads[1:], "k--", linewidth=2, alpha=0.5, label="Ideal (Linear)")
ax2.set_title("Speedup vs Thread Count", fontsize=13, fontweight="bold")
ax2.set_xlabel("Number of Threads")
ax2.set_ylabel("Speedup (T1 / Tt)")
ax2.set_xscale("log", base=2)
ax2.set_xticks(threads[1:])
ax2.set_xticklabels(threads[1:])
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3) Throughput (MTEPS) vs threads
ax3 = axes[1, 0]
for _, row in df_top.iterrows():
    m = float(row["num_edges"])
    times = times_for_row(row)

    # Prefer CSV mteps if present; otherwise compute from times.
    if all(c in df.columns for c in mteps_cols) and row[mteps_cols].notna().all():
        mteps = np.array([row[f"mteps_t{t}"] for t in threads], dtype=float)
        label = label_for(row) + " [CSV mteps]"
    else:
        mteps = mteps_from_times(m, times)
        label = label_for(row) + " [recalc]"

    ax3.plot(threads, mteps, marker="^", linewidth=2, markersize=7, label=label)

ax3.set_title("Throughput (MTEPS)", fontsize=13, fontweight="bold")
ax3.set_xlabel("Number of Threads")
ax3.set_ylabel("MTEPS (Million Edges / Second)")
ax3.set_xscale("log", base=2)
ax3.set_xticks(threads)
ax3.set_xticklabels(threads)
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4) Parallel efficiency vs threads
ax4 = axes[1, 1]
all_eff_values = []
for _, row in df_top.iterrows():
    times = times_for_row(row)
    speedups = speedups_from_times(times)
    eff = efficiency_from_speedups(speedups)
    all_eff_values.append(eff[1:])
    ax4.plot(threads[1:], eff[1:], marker="D", linewidth=2, markersize=7, label=label_for(row))

ax4.axhline(100, color="k", linestyle="--", linewidth=2, alpha=0.5, label="Ideal (100%)")
ax4.set_title("Parallel Efficiency", fontsize=13, fontweight="bold")
ax4.set_xlabel("Number of Threads")
ax4.set_ylabel("Efficiency (%) = (Speedup / Threads) × 100")
ax4.set_xscale("log", base=2)
ax4.set_xticks(threads[1:])
ax4.set_xticklabels(threads[1:])
ax4.grid(True, alpha=0.3)

eff_max = float(np.max(np.vstack(all_eff_values))) if all_eff_values else 100.0
ax4.set_ylim(0, max(110.0, eff_max * 1.1))
ax4.legend()

plt.tight_layout()
plt.savefig("bfs_openacc_performance.png", dpi=300, bbox_inches="tight")
print("✓ Saved: bfs_openacc_performance.png")

# -----------------------------
# Summary table image (top 3)
# -----------------------------
fig2, ax = plt.subplots(figsize=(14, 4))
ax.axis("tight")
ax.axis("off")

summary_rows = []
for _, row in df_top.iterrows():
    n = int(row["num_nodes"])
    m = float(row["num_edges"])
    times = times_for_row(row)
    speedups = speedups_from_times(times)

    mteps_1  = mteps_from_times(m, times)[0]
    mteps_32 = mteps_from_times(m, times)[threads.index(32)]

    summary_rows.append([
        row["graph"],
        f"{n:,}",
        f"{int(m):,}",
        f"{row['avg_degree']:.2f}" if "avg_degree" in row and pd.notna(row["avg_degree"]) else "",
        f"{times[0]:.3f}",
        f"{times[threads.index(32)]:.3f}",
        f"{speedups[threads.index(32)]:.2f}×",
        f"{mteps_1:.2f}",
        f"{mteps_32:.2f}",
    ])

table = ax.table(
    cellText=summary_rows,
    colLabels=["Graph", "Nodes", "Edges", "Avg Degree",
               "Time (1T) ms", "Time (32T) ms", "Speedup@32", "MTEPS@1", "MTEPS@32"],
    cellLoc="center",
    loc="center",
    colWidths=[0.16, 0.10, 0.12, 0.10, 0.11, 0.11, 0.11, 0.09, 0.10],
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

for j in range(9):
    table[(0, j)].set_facecolor("#4472C4")
    table[(0, j)].set_text_props(weight="bold", color="white")

for i in range(1, len(summary_rows) + 1):
    if i % 2 == 0:
        for j in range(9):
            table[(i, j)].set_facecolor("#E7E6E6")

plt.title("BFS OpenACC Summary (Top 3 by Speedup@32)", fontsize=14, fontweight="bold", pad=18)
plt.savefig("bfs_openacc_summary_table.png", dpi=300, bbox_inches="tight")
print("✓ Saved: bfs_openacc_summary_table.png")

# -----------------------------
# Console summary
# -----------------------------
print("\n=== Top 3 by Speedup@32 (recomputed from times) ===")
for _, row in df_top.iterrows():
    n = int(row["num_nodes"])
    m = float(row["num_edges"])
    times = times_for_row(row)
    speedups = speedups_from_times(times)
    eff = efficiency_from_speedups(speedups)

    s32 = speedups[threads.index(32)]
    e32 = eff[threads.index(32)]
    mteps_vals = mteps_from_times(m, times)

    print(f"\n{row['graph']}:")
    print(f"  Graph: nodes={n:,}, edges={int(m):,}, avg_degree={float(row['avg_degree']):.3f}")
    print(f"  Time:  T1={times[0]:.6f} ms, T32={times[threads.index(32)]:.6f} ms")
    print(f"  Speedup@32:     {s32:.2f}×")
    print(f"  Efficiency@32:  {e32:.1f}%")
    print(f"  Throughput:     {mteps_vals[0]:.2f} → {mteps_vals[threads.index(32)]:.2f} MTEPS")

plt.show()
