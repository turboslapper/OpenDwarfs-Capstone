import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('bfs_profile.csv')

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Get unique graphs (excluding chesapeake for cleaner visualization)
graphs = ['hollywood-2009', 'ldoor', 'roadNet-CA']

# Filter data for these graphs and take average of multiple runs
df_filtered = df[df['graph'].isin(graphs)]
df_avg = df_filtered.groupby('graph').mean().reset_index()

# Thread counts
threads = [1, 2, 4, 8, 16, 32]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('BFS OpenMP Parallelization Performance Analysis', fontsize=16, fontweight='bold')

# 1. Execution Time vs Thread Count
ax1 = axes[0, 0]
for graph in graphs:
    data = df_avg[df_avg['graph'] == graph]
    times = [data[f'ms_t{t}'].values[0] for t in threads]
    ax1.plot(threads, times, marker='o', linewidth=2, label=graph, markersize=8)

ax1.set_xlabel('Number of Threads', fontsize=12)
ax1.set_ylabel('Execution Time (ms)', fontsize=12)
ax1.set_title('Execution Time vs Thread Count', fontsize=13, fontweight='bold')
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xticks(threads)
ax1.set_xticklabels(threads)

# 2. Speedup vs Thread Count
ax2 = axes[0, 1]
for graph in graphs:
    data = df_avg[df_avg['graph'] == graph]
    speedups = [data[f'speedup_t{t}'].values[0] for t in threads[1:]]
    ax2.plot(threads[1:], speedups, marker='s', linewidth=2, label=graph, markersize=8)

# Add ideal speedup line
ax2.plot(threads[1:], threads[1:], 'k--', linewidth=2, label='Ideal (Linear)', alpha=0.5)

ax2.set_xlabel('Number of Threads', fontsize=12)
ax2.set_ylabel('Speedup', fontsize=12)
ax2.set_title('Speedup vs Thread Count', fontsize=13, fontweight='bold')
ax2.set_xscale('log', base=2)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xticks(threads[1:])
ax2.set_xticklabels(threads[1:])

# 3. Throughput (MTEPS) vs Thread Count
ax3 = axes[1, 0]
for graph in graphs:
    data = df_avg[df_avg['graph'] == graph]
    mteps = [data[f'mteps_t{t}'].values[0] for t in threads]
    ax3.plot(threads, mteps, marker='^', linewidth=2, label=graph, markersize=8)

ax3.set_xlabel('Number of Threads', fontsize=12)
ax3.set_ylabel('Throughput (MTEPS)', fontsize=12)
ax3.set_title('Graph Traversal Throughput', fontsize=13, fontweight='bold')
ax3.set_xscale('log', base=2)
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_xticks(threads)
ax3.set_xticklabels(threads)

# 4. Parallel Efficiency (Speedup / Threads)
ax4 = axes[1, 1]
for graph in graphs:
    data = df_avg[df_avg['graph'] == graph]
    efficiency = [data[f'speedup_t{t}'].values[0] / t * 100 for t in threads[1:]]
    ax4.plot(threads[1:], efficiency, marker='D', linewidth=2, label=graph, markersize=8)

ax4.axhline(y=100, color='k', linestyle='--', linewidth=2, label='Ideal (100%)', alpha=0.5)
ax4.set_xlabel('Number of Threads', fontsize=12)
ax4.set_ylabel('Parallel Efficiency (%)', fontsize=12)
ax4.set_title('Parallel Efficiency', fontsize=13, fontweight='bold')
ax4.set_xscale('log', base=2)
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_xticks(threads[1:])
ax4.set_xticklabels(threads[1:])
ax4.set_ylim(0, 110)

plt.tight_layout()
plt.savefig('bfs_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: bfs_performance.png")

# Create a summary table visualization
fig2, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

# Prepare summary data
summary_data = []
for graph in graphs:
    data = df_avg[df_avg['graph'] == graph]
    summary_data.append([
        graph,
        f"{int(data['num_nodes'].values[0]):,}",
        f"{int(data['num_edges'].values[0]):,}",
        f"{data['ms_t1'].values[0]:.1f}",
        f"{data['ms_t32'].values[0]:.1f}",
        f"{data['speedup_t32'].values[0]:.2f}×",
        f"{data['mteps_t32'].values[0]:.1f}"
    ])

table = ax.table(cellText=summary_data,
                colLabels=['Graph', 'Nodes', 'Edges', 'Time (1T)', 'Time (32T)', 'Speedup', 'MTEPS'],
                cellLoc='center',
                loc='center',
                colWidths=[0.18, 0.12, 0.15, 0.12, 0.12, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style the header
for i in range(7):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data) + 1):
    for j in range(7):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

plt.title('BFS Performance Summary (32 Threads)', fontsize=14, fontweight='bold', pad=20)
plt.savefig('bfs_summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: bfs_summary_table.png")

plt.show()

print("\n=== Performance Summary ===")
for graph in graphs:
    data = df_avg[df_avg['graph'] == graph]
    print(f"\n{graph}:")
    print(f"  Best Speedup: {data['speedup_t32'].values[0]:.2f}× (32 threads)")
    print(f"  Peak Throughput: {data['mteps_t32'].values[0]:.1f} MTEPS")
    print(f"  Time Reduction: {data['ms_t1'].values[0]:.1f}ms → {data['ms_t32'].values[0]:.1f}ms")