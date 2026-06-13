# HNSW-Derived IVF via Multilevel Graph Partitioning

## Motivation

IVF (Inverted File Index) partitions vectors into k clusters for efficient approximate nearest neighbor search. Standard construction runs k-means to find cluster centroids — expensive at scale.

When an HNSW index already exists (as in MemWeaver's hot tier), the graph structure implicitly encodes the same neighborhood information k-means is trying to discover. This document describes how to derive IVF partitions from the existing HNSW structure without running k-means.

---

## Key Insight

**HNSW construction already paid the cost of understanding neighborhood structure. IVF construction should reuse it rather than recompute it.**

This is a communication-avoiding principle — the same insight behind Flash Attention (recompute rather than store/reload) and Demmel's communication-avoiding linear algebra.

---

## HNSW Hierarchy as Approximate Coarsening

HNSW maintains a multilevel graph:

```
Layer 0: 1,000,000 nodes — all vectors, dense neighborhood edges
Layer 1:    60,000 nodes — subset, longer-range connections
Layer 2:     3,750 nodes — smaller subset, even longer range
Layer 3:       234 nodes — very sparse
Layer 4:        15 nodes — near entry point
```

**Important distinction from standard multilevel coarsening:**

Layer l+1 nodes are NOT supernodes of layer l clusters. They are independently selected nodes that happen to be well-distributed in vector space. The parent-child relationship must be constructed explicitly — it does not come free from HNSW.

---

## The Two Components

IVF construction from HNSW requires two separate inputs:

```
Component 1: HNSW hierarchy
  Used for: matching — which vectors group together
  Layer l nodes = natural region representatives
  Vectors assigned to nearest layer l node = their group
  
Component 2: Explicit fine graph
  Used for: coarse graph edge construction
  Edges derived by contracting fine graph
  Cut consistency guaranteed
  
HNSW provides good matching.
Fine graph contraction provides correct coarse graph.
```

---

## The Algorithm

### Phase 1: Build Explicit Fine Graph

```rust
// G0: layer 0 HNSW edges as explicit adjacency list
struct CoarseGraph {
    n_nodes: usize,
    edges: Vec<Vec<(NodeId, f32)>>,  // (neighbor, weight)
}

fn build_fine_graph(hnsw: &HnswArena) -> CoarseGraph {
    let mut edges = vec![vec![]; hnsw.len()];
    
    for node in hnsw.all_nodes() {
        for &neighbor in hnsw.neighbors_at(node, 0) {
            let w = distance(hnsw.vector_at(node), hnsw.vector_at(neighbor));
            edges[node.as_usize()].push((neighbor, w));
        }
    }
    
    CoarseGraph { n_nodes: hnsw.len(), edges }
}
```

### Phase 2: Build Coarsening Hierarchy

For each level l, assign each layer l-1 node to its nearest layer l representative within the same partition:

```rust
fn assign_parent(
    node: NodeId,
    partition: &Partition,
    hnsw: &HnswArena,
    level: usize,
) -> Option<NodeId> {
    // Find nearest layer `level` node in same partition
    hnsw.neighbors_at(node, level - 1)
        .iter()
        .filter(|&&n| hnsw.max_level(n) >= level)
        .filter(|&&n| partition[n] == partition[node])  // same partition only
        .min_by(|&&a, &&b| {
            distance(hnsw.vector_at(node), hnsw.vector_at(a))
                .partial_cmp(&distance(hnsw.vector_at(node), hnsw.vector_at(b)))
                .unwrap()
        })
        .copied()
}
```

**Critical invariant: only merge nodes within the same partition.**

This ensures:
- Coarse graph edges faithfully represent fine graph cut
- Cross-partition edges preserved through coarsening
- Cut consistency maintained at every level

### Phase 3: Derive Coarse Graph by Edge Contraction

```rust
fn coarsen(
    fine: &CoarseGraph,
    matching: &HashMap<NodeId, NodeId>,  // node → supernode
) -> CoarseGraph {
    let mut coarse_edges: HashMap<(NodeId, NodeId), f32> = HashMap::new();
    
    for (u, neighbors) in fine.edges.iter().enumerate() {
        let su = matching[&NodeId(u as u32)];
        
        for &(v, weight) in neighbors {
            let sv = matching[&v];
            
            if su != sv {
                // Edge crosses supernode boundary — preserve in coarse graph
                let key = (su.min(sv), su.max(sv));
                *coarse_edges.entry(key).or_insert(0.0) += weight;
            }
            // su == sv: edge collapses — both in same supernode
        }
    }
    
    CoarseGraph::from_edges(coarse_edges)
}
```

**Cut consistency guarantee:**

```
For each fine edge (u, v):
  Internal edge (same partition):
    su == sv: collapses — was internal, no cut contribution
    su != sv: preserved as (su, sv) internal — cut unchanged
    
  Cut edge (different partitions):
    su in P, sv in Q ≠ P (by same-partition matching invariant)
    Preserved as (su, sv) cut edge
    
cut(coarse) == cut(fine) exactly
```

### Phase 4: Initial Partition at Coarsest Viable Level

For k=256 partitions, layer 2 with ~3,750 nodes is the starting point:

```
Level 2 (~3,750 nodes): small enough for fast exact partitioning
                         large enough for meaningful structure
                         
Level 3 (~234 nodes):   too few nodes for k=256 partitions
Level 1 (~60k nodes):   works but slower initial partition
```

Use KaHyPar or similar for the initial partition at level 2:

```bash
KaHyPar -h graph.hgr -k 256 -e 0.03 -o cut -m direct
```

### Phase 5: Uncoarsening with KL/FM Refinement

Project partition down through levels, refining at each step.

**The KL/FM algorithm:**

At each level, apply Fiduccia-Mattheyses:

1. Make moves greedily — always pick highest gain unlocked node
2. Lock each moved node — cannot move again this pass
3. Record every move and cumulative gain
4. Stop when all nodes locked (local optimum)
5. Find prefix with highest cumulative gain
6. Revert to that point — rollback moves after best prefix

```
Move sequence: m1, m2, m3, m4, m5
Individual gains: -2, +5, -1, +3, -4
Cumulative:       -2, +3, +2, +5, +1
                              ^
Best prefix: moves 1-4, net gain = +5
Accept these, revert m5
```

**Key property: start delta at 0, accept any sequence with positive net gain.**

No need for exact global cut values. Just track whether each pass produces net improvement.

```rust
fn fm_pass(graph: &CoarseGraph, partition: &mut Partition) -> i64 {
    let mut locked = vec![false; graph.n_nodes];
    let mut moves: Vec<(NodeId, PartitionId, PartitionId)> = vec![];
    let mut cumulative_gains: Vec<i64> = vec![];
    let mut running_gain = 0i64;
    
    loop {
        // Find best unlocked move via gain bucket
        let Some((node, target, gain)) = gain_bucket.pop_max_unlocked() else {
            break;
        };
        
        let from = partition[node];
        execute_move(node, from, target, partition);
        locked[node.as_usize()] = true;
        
        running_gain += gain;
        moves.push((node, from, target));
        cumulative_gains.push(running_gain);
    }
    
    // Find best prefix
    let (best_prefix, best_gain) = cumulative_gains
        .iter()
        .enumerate()
        .max_by_key(|(_, &g)| g)
        .unwrap_or((0, &0));
    
    // Revert moves after best prefix
    for (node, from, _to) in moves[best_prefix + 1..].iter().rev() {
        partition[*node] = *from;
    }
    
    *best_gain
}

fn kl_refine(graph: &CoarseGraph, partition: &mut Partition) {
    loop {
        let gain = fm_pass(graph, partition);
        if gain <= 0 { break; }
    }
}
```

**Cut update during KL move:**

When node u moves from partition A to partition B:

```
For each neighbor v of u:
  v in A: was internal → now cut    → cut += 1
  v in B: was cut     → now internal → cut -= 1
  v in C: cut edge shifts A→B       → cut unchanged

Net cut change = internal_degree[u] - external_degree[u][B]
               = -gain(u, A→B)
```

**Gain formula:**

```
gain(u, A→B) = external_degree[u][B] - internal_degree[u]

Where:
  external_degree[u][B] = edges from u to partition B (gained)
  internal_degree[u]    = edges from u to partition A (lost)
```

### Phase 6: Project Partition to Layer 0

After KL refinement at level 2:

```
Level 2 partition → project to level 1:
  Each level 1 node inherits parent's partition from level 2
  KL refinement at level 1
  
Level 1 partition → project to level 0:
  Each level 0 node inherits parent's partition from level 1
  KL refinement at level 0 (final)
```

```rust
fn project_down(
    coarse_partition: &Partition,
    matching: &HashMap<NodeId, NodeId>,
    fine_n: usize,
) -> Partition {
    let mut fine_partition = vec![0usize; fine_n];
    
    for (fine_node, &supernode) in matching.iter() {
        fine_partition[fine_node.as_usize()] = coarse_partition[supernode];
    }
    
    fine_partition
}
```

---

## Build Time Estimate

```
Fine graph construction:    O(n × M_MAX0) = O(32M) ≈ 2s
Coarsening hierarchy:       O(n × log n) ≈ 3s
Initial partition (level 2): milliseconds (3,750 nodes)
KL at level 2:               milliseconds
KL at level 1:               ~1s (60k nodes)
KL at level 0:               ~5s (1M nodes, few boundary nodes)

Total: ~10-15s
vs k-means: ~25-45s

Speedup: ~2-3x
Additional benefit: better recall from neighborhood-coherent partitions
```

---

## Recall Quality Argument

K-means minimizes within-cluster variance — a proxy for recall.

HNSW-derived IVF minimizes neighborhood cuts directly — vectors whose true nearest neighbors are in different partitions. This is a more direct optimization for recall.

```
K-means IVF:
  Partition boundary = Voronoi boundary
  May split natural neighborhoods
  
HNSW-derived IVF:
  Partition boundary = graph cut boundary
  Respects HNSW neighborhood structure
  Fewer cross-partition true nearest neighbors
  → Higher recall at same nprobe
```

---

## Relationship to Known Techniques

| Concept | Source | Application here |
|---------|--------|-----------------|
| Multilevel partitioning | METIS (Karypis 1995) | Coarsening schedule |
| Hypergraph partitioning | hMETIS, KaHyPar | Per-node neighborhood as hyperedge |
| KL/FM refinement | Kernighan-Lin 1970, Fiduccia-Mattheyses 1982 | Partition improvement at each level |
| Communication avoidance | Demmel et al. 2012 | Reuse HNSW structure, avoid recomputing |
| HNSW hierarchy | Malkov 2018 | Coarsening candidates |

The contribution: connecting these established techniques to IVF construction for the specific case where HNSW is already built.

---

## Limitations

**Only valuable when HNSW already exists:**

```
Just want IVF (no HNSW):
  k-means: 25-45s
  HNSW build + HNSW-IVF: 84s + 15s = 99s — slower
  → Use k-means
  
HNSW already built (MemWeaver):
  k-means: 25-45s additional
  HNSW-IVF: 15s — faster
  → Use HNSW-IVF
```

**Boundary recall loss:**

Vectors near partition boundaries may miss cross-partition true nearest neighbors. Small effect (~5-10% of vectors), acceptable since HNSW is approximate.

**Not suitable for billion-scale:**

Single-machine HNSW at 1B vectors is impractical. At that scale, distributed k-means (Spark MLlib) is the correct tool regardless.

---

## Implementation Plan

```
Phase 1 (prototype — Go):
  Build fine graph from HNSW layer 0
  Simple FM pass — naive O(n²) for correctness validation
  Test on SIFT1M subset (100k vectors)
  Compare recall vs k-means baseline
  
Phase 2 (optimization):
  Gain bucket for O(n × degree) FM passes
  Full SIFT1M (1M vectors)
  Tune: starting level, number of passes, balance epsilon
  
Phase 3 (integration):
  Background IVF converter in MemWeaver
  Converts sealed arena files to IVF partitions
  Uploads per-cluster HNSW files to S3
  Reader nodes cache and serve
```

---

## Summary

```
Input:   existing HNSW index (already built for hot tier)
Output:  IVF partitions respecting neighborhood structure

Method:
  1. Extract fine graph from HNSW layer 0 edges
  2. Use HNSW upper layers to guide matching (within partition)
  3. Contract fine graph to build coarse graph (cut consistent)
  4. Partition coarse graph at level 2 (fast, small)
  5. Uncoarsen with KL/FM refinement at each level
  6. Final IVF partition at layer 0

Cost:    ~10-15s (vs 25-45s for k-means)
Quality: comparable or better recall (neighborhood-coherent)
Caveat:  only worthwhile when HNSW already built
```
