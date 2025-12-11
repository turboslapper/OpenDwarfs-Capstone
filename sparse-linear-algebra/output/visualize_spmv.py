import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('spmv_profile.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Sort by best 32-thread speedup and keep the top 3 configurations
df_sorted = df.sort_values('speedup_t32', ascending=False).reset_index(drop=True)
df_top = df_sorted.head(3)

# Thread counts
threads = [1, 2, 4, 8, 16, 32]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SpMV OpenMP Performance: Top 3 Configurations', fontsize=16, fontweight='bold')

# 1. Execution Time vs Thread Count
ax1 = axes[0, 0]
for _, row in df_top.iterrows():
    label = f"{row['graph']} (speedup_32 = {row['speedup_t32']:.2f}x)"
    times = [row[f'ms_t{t}'] for t in threads]
    ax1.plot(threads, times, marker='o', linewidth=2, label=label, markersize=8)

ax1.set_xlabel('Number of Threads', fontsize=12)
ax1.set_ylabel('Execution Time (ms)', fontsize=12)
ax1.set_title('Execution Time vs Thread Count (Top 3)', fontsize=13, fontweight='bold')
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xticks(threads)
ax1.set_xticklabels(threads)

# 2. Speedup vs Thread Count
ax2 = axes[0, 1]
for _, row in df_top.iterrows():
    label = f"{row['graph']} (speedup_32 = {row['speedup_t32']:.2f}x)"
    speedups = [row[f'speedup_t{t}'] for t in threads[1:]]
    ax2.plot(threads[1:], speedups, marker='s', linewidth=2, label=label, markersize=8)

# Add ideal speedup line
ax2.plot(threads[1:], threads[1:], 'k--', linewidth=2, label='Ideal (Linear)', alpha=0.5)

ax2.set_xlabel('Number of Threads', fontsize=12)
ax2.set_ylabel('Speedup', fontsize=12)
ax2.set_title('Speedup vs Thread Count (Top 3)', fontsize=13, fontweight='bold')
ax2.set_xscale('log', base=2)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xticks(threads[1:])
ax2.set_xticklabels(threads[1:])

# 3. GFLOPS (estimated) vs Thread Count
ax3 = axes[1, 0]
for _, row in df_top.iterrows():
    label = f"{row['graph']} (speedup_32 = {row['speedup_t32']:.2f}x)"
    nnz = row['nnz']
    # SpMV: 2 * nnz operations (multiply + add per non-zero)
    gflops = [(2.0 * nnz / (row[f'ms_t{t}'] * 1e6)) for t in threads]
    ax3.plot(threads, gflops, marker='^', linewidth=2, label=label, markersize=8)

ax3.set_xlabel('Number of Threads', fontsize=12)
ax3.set_ylabel('Performance (GFLOPS)', fontsize=12)
ax3.set_title('SpMV Computational Throughput (Top 3)', fontsize=13, fontweight='bold')
ax3.set_xscale('log', base=2)
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_xticks(threads)
ax3.set_xticklabels(threads)

# 4. Parallel Efficiency (Speedup / Threads)
ax4 = axes[1, 1]
for _, row in df_top.iterrows():
    label = f"{row['graph']} (speedup_32 = {row['speedup_t32']:.2f}x)"
    efficiency = [row[f'speedup_t{t}'] / t * 100 for t in threads[1:]]
    ax4.plot(threads[1:], efficiency, marker='D', linewidth=2, label=label, markersize=8)

ax4.axhline(y=100, color='k', linestyle='--', linewidth=2, label='Ideal (100%)', alpha=0.5)
ax4.set_xlabel('Number of Threads', fontsize=12)
ax4.set_ylabel('Parallel Efficiency (%)', fontsize=12)
ax4.set_title('Parallel Efficiency (Top 3)', fontsize=13, fontweight='bold')
ax4.set_xscale('log', base=2)
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_xticks(threads[1:])
ax4.set_xticklabels(threads[1:])
ax4.set_ylim(0, 110)

plt.tight_layout()
plt.savefig('spmv_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: spmv_performance.png")

# Create a summary table visualization
fig2, ax = plt.subplots(figsize=(14, 4))
ax.axis('tight')
ax.axis('off')

# Prepare summary data for the same top 3 configurations
summary_data = []
for _, row in df_top.iterrows():
    nnz = row['nnz']
    gflops_32 = 2.0 * nnz / (row['ms_t32'] * 1e6)
    summary_data.append([
        row['graph'],
        f"{int(row['num_rows']):,}",
        f"{int(nnz):,}",
        f"{row['avg_degree']:.2f}",
        f"{row['ms_t1']:.2f}",
        f"{row['ms_t32']:.2f}",
        f"{row['speedup_t32']:.2f}×",
        f"{gflops_32:.3f}"
    ])

table = ax.table(cellText=summary_data,
                colLabels=['Graph', 'Rows', 'Non-zeros', 'Avg Degree', 'Time (1T)', 'Time (32T)', 'Speedup', 'GFLOPS'],
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.12, 0.13, 0.11, 0.11, 0.11, 0.11, 0.11])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style the header
for i in range(8):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data) + 1):
    for j in range(8):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

plt.title('SpMV Performance Summary (Top 3 by 32-thread Speedup)', fontsize=14, fontweight='bold', pad=20)
plt.savefig('spmv_summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: spmv_summary_table.png")

plt.show()

print("\n=== SpMV Performance Summary (Top 3 by 32-thread Speedup) ===")
for _, row in df_top.iterrows():
    nnz = row['nnz']
    gflops_1 = 2.0 * nnz / (row['ms_t1'] * 1e6)
    gflops_32 = 2.0 * nnz / (row['ms_t32'] * 1e6)
    print(f"\n{row['graph']}:")
    print(f"  Matrix: {int(row['num_rows']):,} × {int(row['num_cols']):,}, NNZ: {int(nnz):,}")
    print(f"  Best Speedup: {row['speedup_t32']:.2f}× (32 threads)")
    print(f"  Performance: {gflops_1:.3f} → {gflops_32:.3f} GFLOPS")
    print(f"  Time Reduction: {row['ms_t1']:.2f}ms → {row['ms_t32']:.2f}ms")