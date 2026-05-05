# Multi-vector storage (`MultiVectorBlock`)

This crate’s core table type is **`MultiVectorBlock`**: an arena-backed structure that stores **many rows**, where each row can hold **several named vector fields** (e.g. `text`, `image`) with fixed per-field dimensions.

## “Vector” and “multivector” in vector DB search (general)

These terms are used loosely in products and papers; below is a practical split.

**Vector** — In most **vector databases** and **semantic search**, a *vector* is a **single embedding**: a fixed-length array of floats (often hundreds or thousands of dimensions) from an encoder. One chunk, passage, or object is often represented by **one** vector for **similarity search** (cosine, dot product, L2) and **ANN** indexes (HNSW, IVF, etc.). “Vector search” usually means **nearest neighbors in embedding space**.

**Multivector** — Usually **more than one vector per logical item**, in one of these senses:

1. **Late-interaction / ColBERT-style** — A document or query is **many** vectors (e.g. per token or span), not one pooled vector. Scoring uses **interactions between sets** of vectors (e.g. MaxSim), not a single pairwise dot product between two vectors.
2. **Multi-field / multi-representation** — One record has **several named embeddings** (e.g. title, body, image), possibly different dimensions or models. Search may fuse per-field scores or query specific fields.
3. **Index-centric wording** — Some docs say “multi-vector” when the **index** stores **multiple** vectors per key (chunks, versions, or modalities).

**How this crate fits** — `MultiVectorBlock` is mainly the **multi-field per row** idea (sense 2): one row, several `VectorBlock` columns. **ColBERT**-style “many vectors per document” (sense 1) is a different retrieval pattern; this crate still supports **importing** such tensors into a field via the ColBERT helpers on `MultiVectorBlock`, but the *indexing/scoring* story is separate from “one dense vector per row” search.

## What it is

- **`VectorBlock`**: one named field — a column of `f32` vectors of fixed `dim`, plus a parallel `VectorId` per row, SIMD-friendly layout.
- **`MultiVectorBlock`**: one **block** / table — for each field in the schema, a `VectorBlock`; row index `i` is the **same logical record** across all fields (columnar alignment).

Field names, offsets, the `VectorBlock` table, and payload slabs are allocated in a single **`Arena`** (bump allocator over an mmap).

## Naming: “multivector” vs `MultiVectorBlock`

- **`MultiVectorBlock`** means **multi-field storage in one block**, and the block holds **many records** (up to its row `capacity`).
- Informally, **“a multivector”** often means **one** item’s multi-field vectors — that matches **one row** in the block (what you pass to `push_record`), not the whole `MultiVectorBlock`.

So: **one `MultiVectorBlock` = many multi-field rows**, not a single multivector.

## API sketch

- **`MultiVectorBlock::try_new(arena, fields, capacity)`** — build one block with a max row count; allocates from `arena` up front for that capacity. Returns `None` if the arena is too small.
- **`push_record` / `get_record`** — append or read row `i` across all fields.

## Examples

### Vector vs multivector (search scenarios)

| Scenario | What you store | Typical query |
|----------|----------------|---------------|
| **Classic vector search** | One embedding per product image | “Find images nearest this query embedding” (cosine / L2 on **one** vector per row) |
| **Multi-field** | Title embedding (dim 384) + body embedding (dim 768) per document | “Match on title, body, or blend scores” — **several named vectors per row** |
| **ColBERT-style** | Hundreds of token vectors per passage | **MaxSim**-style scoring over **sets** of vectors, not one dot product per document |

### Example: *vector* (one embedding per item)

You index **products** for similarity search. Each product has **one** 512-dimensional embedding from an image model.

- **Stored per row:** `v ∈ ℝ^512` (and an id).
- **Query:** user image → encoder → `q ∈ ℝ^512`.
- **Score:** e.g. `cosine(q, v)` (or L2 / dot product, depending on index).
- **ANN index:** HNSW / IVF over **one vector per row** — “find the k nearest `v` to `q`.”

Nothing here is “multivector”: there is **a single vector** representing the whole item for that retrieval path.

### Example: *multivector* — multiple fields per item

You index **news articles**. Each article has:

- **Title** embedding `t ∈ ℝ^384` (small fast model),
- **Body** embedding `b ∈ ℝ^768` (larger model).

So **one logical document** carries **two vectors** (possibly different dims and models). That is **multivector** in the **multi-field** sense.

- **Query** might produce `q_t`, `q_b` or a single `q` projected per field.
- **Scoring** might be `0.4 · sim(q, t) + 0.6 · sim(q, b)`, or search title and body indexes separately and merge ranks.

This matches how you’d use **two `VectorBlock` columns** in one `MultiVectorBlock` row (`title`, `body`).

### Example: *multivector* — many vectors per passage (ColBERT-style)

A **passage** is not summarized by one pooled vector: you keep **one vector per token** (or per subword), so the passage is a **set** of vectors `{e_1, …, e_N} ⊂ ℝ^d`, and the query is another set `{q_1, …, q_M}`.

- **Score** is **not** a single `dot(e_query, e_doc)` for the whole passage; you use **late interaction** (e.g. **MaxSim**: each query vector finds its best-matching passage vector, then aggregate).
- In storage, a “document” can look like a **matrix** of shape `N × d` (flattened or as a tensor field) — **multivector** in the **late-interaction** sense.

This crate can **store** such tensors in a dedicated field via the ColBERT helpers on `MultiVectorBlock`; **how you score** them at query time is a separate retrieval layer.

### How a multivector is *created* for a passage or document

This is about **encoding** (usually a neural model), not about mmap layout.

**One vector per passage (classic dense retrieval)**  
1. **Tokenize** the passage (and truncate/pack to a max length).  
2. Run a **single encoder** (e.g. sentence Transformer, BERT-style with a pooling head).  
3. **Pool** the sequence into one vector: e.g. `[CLS]` embedding, **mean** over token embeddings, or a dedicated pooling layer.  
4. Optionally **normalize** (L2) and optionally **project** to a smaller dimension.  
→ You get **one** `d`-dimensional vector per passage. That is **not** a multivector for retrieval (it’s a single embedding).

**Multivector — several vectors from different parts or modalities (multi-field)**  
1. **Split** the document into channels: title vs body, or caption vs image.  
2. Run **one forward pass per channel** (possibly **different models** or different max lengths): e.g. `encode(title)`, `encode(body)`.  
3. Each channel yields **one** pooled vector (or one vector per channel as designed).  
→ You get **a small set of named vectors** per document (e.g. `title`, `body`). Creation is: **multiple independent encodings**, then you store them as separate fields.

**Multivector — many vectors per passage (late-interaction / ColBERT-style)**  
1. **Tokenize** to subwords; keep **order** (with special tokens as the model expects).  
2. Run a **transformer** over the full passage.  
3. For **each token position** you care about (e.g. each contextualized token, often excluding only padding), take that position’s **hidden vector** from a chosen layer.  
4. Often apply a **learned linear map** to a smaller dimension `d` and **L2-normalize** each token vector.  
→ You get **N vectors** in `ℝ^d` for a passage with N retained positions — that **set** is the multivector representation. The **query** is encoded the same way, giving **M** query vectors; scoring compares the two **sets** (e.g. MaxSim), not one pooled dot product.

Long documents may be **chunked** or **truncated**; each chunk can produce its own multivector or its own token stack, depending on your indexer.

### `MultiVectorBlock` as a table (conceptual)

Schema: fields `text` (dim 4), `image` (dim 8). One block, row capacity 10 — **column** = one `VectorBlock`, **row** = one logical record.

| row `i` | `text` (4 floats) | `image` (8 floats) |
|--------:|-------------------|---------------------|
| 0 | one vector | one vector |
| 1 | one vector | one vector |
| … | … | … |

Row `i` across fields refers to the **same** `VectorId` / record (see `push_record`).

### Rust: build a block and append one row

```rust
use vector::{Arena, NamedVector, VectorId};
use vector::vector::MultiVectorBlock;

let arena = Arena::with_capacity(4 * 1024 * 1024);
let fields = [("text", 4usize), ("image", 8usize)];
let mut mvb = MultiVectorBlock::try_new(&arena, &fields, 10).unwrap();

let row = vec![
    NamedVector {
        name: "text".into(),
        data: vec![1.0, 0.0, 0.0, 0.0],
    },
    NamedVector {
        name: "image".into(),
        data: vec![0.0; 8],
    },
];

assert!(mvb.push_record(VectorId(42), &row));
let got = mvb.get_record(0);
assert_eq!(got[0].0, "text");
```

Omitted fields in a row are filled with **zeros** for that field’s dimension (see `multi_vector_block_missing_field_is_zeroed` in `vector.rs` tests).

## How `mem_weaver` buckets use it (current design)

In the **`bucket`** crate, each **segment** is typically:

- one **`Arena`** sized to a configured byte budget, and  
- **one `MultiVectorBlock`** for that segment.

So **one arena ↔ one `MultiVectorBlock`** per segment. More rows than fit in that block’s capacity → **another segment** (another arena + another block), not a second block in the same arena.

Inserts **reuse** the same segment until its block is full (by row capacity); a new segment appears on a schedule driven by **`max_rows_per_segment`** and bucket **`capacity`**, not on every single `VectorRecord`.

## Memory and checks

- Row storage for a segment is **mostly allocated when the `MultiVectorBlock` is created** for that `row_cap`; normal **`push_record`** writes into existing slabs and fails when any field’s block is **logically full** (not a per-insert arena byte check).
- Sizing uses helpers like **`arena_bytes_estimate`** / **`max_rows_for_arena_budget`** in `bucket` so construction is intended to fit the arena cap; mismatch can **panic** during `new` if the arena is too small.
- A **large** arena budget with a **small** block, or **many** segments each with its own name/metadata, can **waste** virtual space or duplicate metadata; tuning `arena_segment_size_bytes` and segment layout addresses that.

For API details, see **`vector.rs`** (`MultiVectorBlock`, `VectorBlock`) and **`types.rs`** (`VectorRecord`, `VectorSchema`, etc.).
