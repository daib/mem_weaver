#!/usr/bin/env python3
"""
Plot memory overhead comparison between arena and naive HNSW implementations.

Usage:
    cargo test --release ... 2> arena.txt   # run arena benchmark, capture stderr
    cargo test --release ... 2> naive.txt   # run naive benchmark, capture stderr
    python3 plot_memory.py arena.txt naive.txt
"""

import sys
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def parse_log(filename):
    vectors, rss, peak, insert_ms = [], [], [], []
    with open(filename) as f:
        for line in f:
            m = re.search(
                r'hnsw_insert (\d+)/\d+.*rss_kb=(\d+) peak_rss_kb=(\d+) insert_phase_ms=([\d.]+)',
                line
            )
            if m:
                n      = int(m.group(1))
                r      = int(m.group(2))
                p      = int(m.group(3))
                ms     = float(m.group(4))
                vectors.append(n)
                rss.append(r / 1024)        # MB
                peak.append(p / 1024)       # MB
                insert_ms.append(ms / 1000) # seconds
    overhead = [p - r for p, r in zip(peak, rss)]
    return vectors, rss, peak, overhead, insert_ms


def find_spikes(vectors, overhead, threshold_mb=15):
    """Return (index, vector_count, overhead_mb) for notable spikes."""
    spikes = []
    for i in range(1, len(overhead)):
        jump = overhead[i] - overhead[i - 1]
        if jump > threshold_mb:
            spikes.append((i, vectors[i], overhead[i]))
    return spikes


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 plot_memory.py <arena_log> <naive_log>")
        sys.exit(1)

    arena_file = sys.argv[1]
    naive_file  = sys.argv[2]

    av, ar, ap, ao, ams = parse_log(arena_file)
    nv, nr, np_, no, nms = parse_log(naive_file)

    arena_build_s = ams[-1] if ams else 0
    naive_build_s = nms[-1] if nms else 0

    naive_spikes = find_spikes(nv, no)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "MemWeaver Arena vs Naive — 1M vectors, dim=128, M=16",
        fontsize=13, fontweight="bold", y=1.02
    )

    # ── Left: peak overhead ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot([v / 1e6 for v in av], ao, color="#0F6E56", linewidth=2, label="Arena")
    ax.plot([v / 1e6 for v in nv], no, color="#993C1D", linewidth=2,
            linestyle="--", label="Naive")

    for _, vec, oh in naive_spikes:
        ax.axvline(x=vec / 1e6, color="#993C1D", linewidth=0.8,
                   linestyle=":", alpha=0.6)
        ax.annotate(
            f"Vec realloc\n{oh:.0f} MB",
            xy=(vec / 1e6, oh),
            xytext=(vec / 1e6 + 0.03, oh + 2),
            fontsize=8, color="#993C1D",
            arrowprops=dict(arrowstyle="->", color="#993C1D", lw=0.8)
        )

    ax.set_xlabel("Vectors inserted (millions)", fontsize=10)
    ax.set_ylabel("Peak − RSS overhead (MB)", fontsize=10)
    ax.set_title("Memory overhead during index build", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Right: RSS growth ────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot([v / 1e6 for v in av], [r - ar[0] for r in ar],
             color="#0F6E56", linewidth=2, label="Arena")
    ax2.plot([v / 1e6 for v in nv], [r - nr[0] for r in nr],
             color="#993C1D", linewidth=2, linestyle="--", label="Naive")

    ax2.set_xlabel("Vectors inserted (millions)", fontsize=10)
    ax2.set_ylabel("RSS growth from baseline (MB)", fontsize=10)
    ax2.set_title("RSS memory growth", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(bottom=0)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Summary stats box ────────────────────────────────────────────────────
    speedup = naive_build_s / arena_build_s if arena_build_s > 0 else 0
    arena_final_overhead = ao[-1] if ao else 0
    naive_final_overhead = no[-1] if no else 0
    arena_rss_growth = ar[-1] - ar[0] if ar else 0
    naive_rss_growth = nr[-1] - nr[0] if nr else 0

    stats = (
        f"Build time:  Arena {arena_build_s:.0f}s  |  Naive {naive_build_s:.0f}s  "
        f"→  {speedup:.2f}x speedup\n"
        f"Final RSS:   Arena +{arena_rss_growth:.0f} MB  |  Naive +{naive_rss_growth:.0f} MB\n"
        f"Peak overhead at 1M:  Arena {arena_final_overhead:.0f} MB  |  "
        f"Naive {naive_final_overhead:.0f} MB\n"
        f"Vec realloc spikes:  {len(naive_spikes)} detected in naive"
    )
    fig.text(0.5, -0.04, stats, ha="center", fontsize=9,
             color="#444", family="monospace")

    plt.tight_layout()
    out = "memweaver_memory_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    print("\n=== Summary ===")
    print(stats)


if __name__ == "__main__":
    main()
