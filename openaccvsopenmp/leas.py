import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11
})

THREADS = [1, 2, 4, 8, 16, 32]

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # numeric conversions (igno cols gracefully)
    numeric_cols = [c for c in df.columns if c.startswith("ms_t") or c.startswith("mteps_t") or c.startswith("speedup_t")]
    numeric_cols += ["num_nodes", "num_edges", "avg_degree", "num_rows", "num_cols", "nnz"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["graph"] = df["graph"].astype(str).str.strip()
    return df

def recompute_speedup32(df: pd.DataFrame) -> pd.Series:
    # source of truth: times
    t1 = df["ms_t1"]
    t32 = df["ms_t32"]
    return t1 / t32

def recompute_mteps32(df: pd.DataFrame) -> pd.Series:
    # MTEPS = (edges / seconds) / 1e6; seconds = ms/1000
    m = df["num_edges"].astype(float)
    t32 = df["ms_t32"].astype(float)
    return (m / (t32 / 1000.0)) / 1e6

def recompute_gflops32(df: pd.DataFrame) -> pd.Series:
    # GFLOPS = (2*nnz ops) / seconds / 1e9
    # seconds = ms/1000, so GFLOPS = (2*nnz) / (ms*1e-3) / 1e9 = (2*nnz)/(ms*1e6)
    nnz = df["nnz"].astype(float)
    t32 = df["ms_t32"].astype(float)
    return (2.0 * nnz) / (t32 * 1e6)

def merge_pair(df_openmp: pd.DataFrame, df_openacc: pd.DataFrame) -> pd.DataFrame:
    # inner join on graph (only compare common graphs)
    m = df_openmp.merge(df_openacc, on="graph", how="inner", suffixes=("_openmp", "_openacc"))
    # keep a stable, readable order by size
    size_col = "num_edges_openmp" if "num_edges_openmp" in m.columns else None
    if size_col:
        m = m.sort_values(size_col, ascending=True).reset_index(drop=True)
    else:
        m = m.sort_values("graph").reset_index(drop=True)
    return m

def grouped_bar(xlabels, y1, y2, title, ylabel, outpng, legend=("OpenMP", "OpenACC")):
    x = np.arange(len(xlabels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width/2, y1, width, label=legend[0], color="#7FB3D5")
    ax.bar(x + width/2, y2, width, label=legend[1], color="#154360")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Graph")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    # annotate bars
    def annotate(vals, offset):
        for i, v in enumerate(vals):
            if np.isnan(v): 
                continue
            ax.text(i + offset, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9, rotation=0)

    annotate(y1, -width/2)
    annotate(y2,  width/2)

    fig.tight_layout()
    fig.savefig(outpng, bbox_inches="tight")
    print(f"✓ Saved: {outpng}")
    plt.close(fig)

def main():
    # ---- BFS
    bfs_mp  = load_csv("bfs_profile_openmp.csv")
    bfs_acc = load_csv("bfs_profile_openacc.csv")

    bfs_mp["speedup32_recalc"]  = recompute_speedup32(bfs_mp)
    bfs_acc["speedup32_recalc"] = recompute_speedup32(bfs_acc)

    bfs_mp["mteps32_recalc"]  = recompute_mteps32(bfs_mp)
    bfs_acc["mteps32_recalc"] = recompute_mteps32(bfs_acc)

    bfs = merge_pair(bfs_mp, bfs_acc)

    grouped_bar(
        xlabels=bfs["graph"].tolist(),
        y1=bfs["speedup32_recalc_openmp"].to_numpy(),
        y2=bfs["speedup32_recalc_openacc"].to_numpy(),
        title="BFS: OpenMP vs OpenACC (Speedup @ 32)",
        ylabel="Speedup (T1 / T32)",
        outpng="bfs_speedup32_openmp_vs_openacc.png",
    )

    grouped_bar(
        xlabels=bfs["graph"].tolist(),
        y1=bfs["mteps32_recalc_openmp"].to_numpy(),
        y2=bfs["mteps32_recalc_openacc"].to_numpy(),
        title="BFS: OpenMP vs OpenACC (Throughput @ 32)",
        ylabel="MTEPS @ 32 (Million edges / sec)",
        outpng="bfs_mteps32_openmp_vs_openacc.png",
    )

    # ---- SpMV
    spmv_mp  = load_csv("spmv_profile_openmp.csv")
    spmv_acc = load_csv("spmv_profile_openacc.csv")

    spmv_mp["speedup32_recalc"]  = recompute_speedup32(spmv_mp)
    spmv_acc["speedup32_recalc"] = recompute_speedup32(spmv_acc)

    # For SpMV, show GFLOPS@32 (needs nnz)
    if "nnz" not in spmv_mp.columns or "nnz" not in spmv_acc.columns:
        raise SystemExit("SpMV CSVs must include 'nnz' to compute GFLOPS. Your spmv_profile.csv should have nnz.")

    spmv_mp["gflops32_recalc"]  = recompute_gflops32(spmv_mp)
    spmv_acc["gflops32_recalc"] = recompute_gflops32(spmv_acc)

    spmv = merge_pair(spmv_mp, spmv_acc)

    grouped_bar(
        xlabels=spmv["graph"].tolist(),
        y1=spmv["speedup32_recalc_openmp"].to_numpy(),
        y2=spmv["speedup32_recalc_openacc"].to_numpy(),
        title="SpMV: OpenMP vs OpenACC (Speedup @ 32)",
        ylabel="Speedup (T1 / T32)",
        outpng="spmv_speedup32_openmp_vs_openacc.png",
    )

    grouped_bar(
        xlabels=spmv["graph"].tolist(),
        y1=spmv["gflops32_recalc_openmp"].to_numpy(),
        y2=spmv["gflops32_recalc_openacc"].to_numpy(),
        title="SpMV: OpenMP vs OpenACC (GFLOPS @ 32)",
        ylabel="GFLOPS @ 32 (2*nnz ops / sec)",
        outpng="spmv_gflops32_openmp_vs_openacc.png",
    )

if __name__ == "__main__":
    main()