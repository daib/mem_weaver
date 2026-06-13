use crate::hnsw::store::{HnswVectorStore, NaiveVectorStore};
pub use common::types::NodeId;
use common::DEFAULT_ARENA_CAPACITY;
use crc32fast::Hasher as Crc32Hasher;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::mem::{align_of, size_of};
use std::path::{Path, PathBuf};
use vector::Arena;

/// Alignment used for arena-backed node storage (`try_alloc_slice_aligned`).
pub const DEFAULT_ALIGNMENT: usize = 8;
// ── Heap-allocated graph (naive) + trait ───────────────────────────────────

/// One vertex: per-level neighbor lists (same layout as the historical `Vec<Node>` implementation).
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub neighbors: Vec<Vec<NodeId>>,
}

impl GraphNode {
    pub fn new(max_level: usize) -> Self {
        Self {
            neighbors: vec![Vec::new(); max_level + 1],
        }
    }

    /// Highest allocated level (`neighbors.len() - 1`). Level `l` uses `neighbors[l]`.
    #[inline]
    pub fn max_level(&self) -> usize {
        self.neighbors.len().saturating_sub(1)
    }

    pub fn ensure_level(&mut self, level: usize) {
        while self.neighbors.len() <= level {
            self.neighbors.push(Vec::new());
        }
    }

    pub fn neighbors_at(&self, level: usize) -> &[NodeId] {
        self.neighbors
            .get(level)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

/// Heap-backed `Vec<GraphNode>`: fully naive graph storage (no arena).
#[derive(Debug, Clone)]
pub struct NaiveNodeStore {
    m: usize,
    m_max0: usize,
    pub(crate) nodes: Vec<GraphNode>,
    pub(crate) vector_store: NaiveVectorStore,
}

impl NaiveNodeStore {
    pub(crate) fn new(m: usize, m_max0: usize) -> Self {
        Self {
            m,
            m_max0,
            nodes: Vec::new(),
            vector_store: NaiveVectorStore::default(),
        }
    }
}

// ── Arena strided blocks + store (VectorStore-style) ─────────────────────────

const MAX_LEVEL: usize = 32;
const LEVELS: usize = MAX_LEVEL + 1;
pub const INVALID_NODE_ID: NodeId = NodeId(u32::MAX);

/// Positional `read_exact` on a `&File` (no cursor mutation). Wraps the platform-specific
/// `FileExt::read_exact_at` on Unix and `seek_read`-style emulation elsewhere.
#[inline]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut remaining = buf;
        let mut off = offset;
        while !remaining.is_empty() {
            let n = file.seek_read(remaining, off)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "read_at hit EOF before filling buffer",
                ));
            }
            remaining = &mut remaining[n..];
            off += n as u64;
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "positional read not supported on this platform",
        ))
    }
}

// the layout of the node is:
// - vector: f32[dim]
// - edges: NodeId[edge_count]
pub struct Node;

impl Node {
    /// Byte size of `f32[dim]` plus padding so `max_level: usize` is aligned.
    #[inline]
    fn vector_span(dim: usize) -> usize {
        (dim * size_of::<f32>()).next_multiple_of(align_of::<usize>())
    }

    /// Byte offset from node base to the start of the packed edge array.
    #[inline]
    fn edges_byte_offset(dim: usize) -> usize {
        Self::vector_span(dim)
    }

    #[inline]
    unsafe fn vector<'a>(node_address: *mut u8, dim: usize) -> &'a mut [f32] {
        std::slice::from_raw_parts_mut(node_address.cast::<f32>(), dim)
    }

    /// Total `NodeId` slots: level `0` uses `m_max0`, each level `1..=max_level` uses `m`.
    #[inline]
    const fn edge_count(max_level: usize, m: usize, m_max0: usize) -> usize {
        m_max0 + max_level * m
    }

    #[inline]
    fn total_size(dim: usize, max_level: usize, m: usize, m_max0: usize) -> usize {
        Self::edges_byte_offset(dim)
            + Self::edge_count(max_level, m, m_max0) * size_of::<NodeId>()
            + 1
    }

    #[inline]
    unsafe fn edges<'a>(
        node_address: *mut u8,
        dim: usize,
        max_level: usize,
        m: usize,
        m_max0: usize,
    ) -> &'a mut [NodeId] {
        std::slice::from_raw_parts_mut(
            node_address
                .add(Self::edges_byte_offset(dim))
                .cast::<NodeId>(),
            Self::edge_count(max_level, m, m_max0),
        )
    }

    #[inline]
    unsafe fn edges_at_level<'a>(
        node_address: *const u8,
        dim: usize,
        level: usize,
        m: usize,
        m_max0: usize,
    ) -> &'a [NodeId] {
        let edge_offset = if level == 0 {
            0
        } else {
            m_max0 + (level - 1) * m
        };
        let cap = if level == 0 { m_max0 } else { m };
        std::slice::from_raw_parts(
            node_address
                .add(Self::edges_byte_offset(dim))
                .wrapping_add(edge_offset * size_of::<NodeId>())
                .cast::<NodeId>(),
            cap,
        )
    }

    #[inline]
    unsafe fn edges_at_level_mut<'a>(
        node_address: *mut u8,
        dim: usize,
        level: usize,
        m: usize,
        m_max0: usize,
    ) -> &'a mut [NodeId] {
        let edge_offset = if level == 0 {
            0
        } else {
            m_max0 + (level - 1) * m
        };
        let cap = if level == 0 { m_max0 } else { m };
        std::slice::from_raw_parts_mut(
            node_address
                .add(Self::edges_byte_offset(dim))
                .wrapping_add(edge_offset * size_of::<NodeId>())
                .cast::<NodeId>(),
            cap,
        )
    }
}

enum NodeBlockStorage {
    InMemory(Arena),
    OnDisk(File),
    /// No local copy: the bytes were evicted via [`NodeBlock::evict`] and live only in
    /// remote storage (e.g. uploaded to S3). Reads (`vector_at`, `neighbors_at`) will
    /// panic until [`NodeBlock::swap_in_from`] restores the block from a path.
    Evicted,
}

impl NodeBlockStorage {
    /// Base pointer of the in-memory mapping. Returns `null` for non-memory variants
    /// (callers must `swap_in()` / `swap_in_from()` before reads).
    #[inline]
    fn as_ptr(&self) -> *const u8 {
        match self {
            NodeBlockStorage::InMemory(a) => a.as_ptr(),
            NodeBlockStorage::OnDisk(_) | NodeBlockStorage::Evicted => std::ptr::null(),
        }
    }
}

pub struct NodeBlock {
    storage: NodeBlockStorage,
    block_index: usize,
    len: usize,
    dim: usize,    // vector dimension
    m: usize,      // target max degree on levels > 0
    m_max0: usize, // target max degree on level 0
}

// SAFETY: The mmap pointer inside NodeBlockStorage is read-only during search.
// Mutations (insert, swap_out, swap_in) always happen under an exclusive write lock,
// so shared read access across threads is safe.
unsafe impl Sync for NodeBlock {}

impl NodeBlock {
    pub fn try_new(dim: usize, m: usize, m_max0: usize, block_index: usize) -> Option<Self> {
        Some(Self {
            storage: NodeBlockStorage::InMemory(
                Arena::try_with_capacity(DEFAULT_ARENA_CAPACITY).unwrap(),
            ),
            len: 0,
            block_index,
            dim,
            m,
            m_max0,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when the block's arena is mapped in RAM and ready for reads/pushes.
    #[inline]
    pub fn is_in_memory(&self) -> bool {
        matches!(self.storage, NodeBlockStorage::InMemory(_))
    }

    /// `true` when the block has been swapped to disk — read APIs will panic until
    /// [`NodeBlock::swap_in`] restores it.
    #[inline]
    pub fn is_on_disk(&self) -> bool {
        matches!(self.storage, NodeBlockStorage::OnDisk(_))
    }

    /// `true` when the block has been [`NodeBlock::evict`]ed — no local arena, no
    /// open fd. Reads panic; [`NodeBlock::swap_in_from`] must restore it from a path.
    #[inline]
    pub fn is_evicted(&self) -> bool {
        matches!(self.storage, NodeBlockStorage::Evicted)
    }

    #[inline]
    fn calculate_node_id(&mut self, offset: usize) -> NodeId {
        // assume that we align to 8 bytes
        // TODO: Make this configurable based on the arena size and the number of bits to ignore for alignment
        // At the moment: 2MB arena = 2^21. Ignore 3 last bits for alignment arrive at 18 bits for offset.
        // The remaining is for node_index
        NodeId((self.block_index << 18 | ((offset >> 3) & ((1 << 18) - 1))) as u32)
    }

    #[inline]
    fn derive_block_index(node_id: NodeId) -> usize {
        (node_id.0 >> 18) as usize
    }

    #[inline]
    fn derive_node_offset(node_id: NodeId) -> usize {
        ((node_id.0 & ((1 << 18) - 1)) as usize) << 3
    }

    // return the index of the new node in the block
    pub fn push_node(&mut self, vector: &[f32], max_level: usize) -> Option<NodeId> {
        let total = Node::total_size(self.dim, max_level, self.m, self.m_max0);
        // Only the in-memory variant supports allocations; swapped-out / evicted blocks
        // are sealed.
        let arena = match &mut self.storage {
            NodeBlockStorage::InMemory(a) => a,
            NodeBlockStorage::OnDisk(_) | NodeBlockStorage::Evicted => return None,
        };
        // we need to align by 8 bytes at the moment
        let node_storage = arena.try_alloc_slice_aligned::<u8>(total, DEFAULT_ALIGNMENT)?;
        let arena_base = arena.as_ptr() as usize;
        let node_offset = node_storage.as_ptr() as usize - arena_base;
        let p = node_storage.as_mut_ptr();
        unsafe {
            Node::vector(p, self.dim).copy_from_slice(vector);
            // *Node::max_level_mut(p, self.dim) = max_level;
            Node::edges(p, self.dim, max_level, self.m, self.m_max0).fill(INVALID_NODE_ID);
        }

        // add the node to the block
        self.len += 1;
        Some(self.calculate_node_id(node_offset))
    }

    #[inline]
    fn calculate_node_address(&self, node_id: NodeId) -> *const u8 {
        let node_offset = NodeBlock::derive_node_offset(node_id);
        self.storage.as_ptr().wrapping_add(node_offset) as *const u8
    }

    /// Read the vector for `node_id`. In-memory blocks return a slice borrowed from the
    /// arena (zero copy). On-disk blocks read `dim * size_of::<f32>()` bytes at the node's
    /// byte offset into `buf` and return a slice borrowed from `buf`.
    pub fn vector_at<'a>(&'a self, node_id: NodeId, buf: &'a mut Vec<u8>) -> &'a [f32] {
        match &self.storage {
            NodeBlockStorage::Evicted => {
                panic!("vector_at on an evicted NodeBlock; call swap_in_from(path) first")
            }
            NodeBlockStorage::InMemory(_) => {
                let node_address = self.calculate_node_address(node_id);
                // SAFETY: arena pointer is valid for the node's full extent; vector starts
                // at the node base.
                unsafe { std::slice::from_raw_parts(node_address.cast::<f32>(), self.dim) }
            }
            NodeBlockStorage::OnDisk(file) => {
                let bytes = self.dim * size_of::<f32>();
                buf.resize(bytes, 0);
                let offset = NodeBlock::derive_node_offset(node_id) as u64;
                read_at(file, buf.as_mut_slice(), offset)
                    .expect("read vector from on-disk node block");
                // SAFETY: `Vec<u8>` returns at least `align_of::<usize>()`-aligned memory
                // on every supported allocator (which is ≥ align_of::<f32>() = 4). Debug
                // builds verify; release builds rely on the allocator's contract.
                debug_assert_eq!(
                    buf.as_ptr() as usize % align_of::<f32>(),
                    0,
                    "Vec<u8> allocation must be f32-aligned"
                );
                unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, self.dim) }
            }
        }
    }

    // get the neighbors of the node at the given level
    pub fn neighbors_at(&self, node_id: NodeId, level: usize, buf: &mut Vec<u8>) -> &[NodeId] {
        assert!(level < LEVELS);
        match &self.storage {
            NodeBlockStorage::Evicted => {
                panic!("neighbors_at on an evicted NodeBlock; call swap_in_from(path) first")
            }
            NodeBlockStorage::InMemory(_) => {
                let node_address = self.calculate_node_address(node_id);
                unsafe { Node::edges_at_level(node_address, self.dim, level, self.m, self.m_max0) }
            }
            NodeBlockStorage::OnDisk(file) => {
                let cap = if level == 0 { self.m_max0 } else { self.m };
                let edge_bytes = cap * size_of::<NodeId>();
                buf.resize(edge_bytes, 0);
                // Absolute byte offset within the file: <node base> + <edges header> +
                // <level offset within edge slab>.
                let edges_within_node = Node::edges_byte_offset(self.dim)
                    + if level == 0 {
                        0
                    } else {
                        (self.m_max0 + (level - 1) * self.m) * size_of::<NodeId>()
                    };
                let abs_offset =
                    (NodeBlock::derive_node_offset(node_id) + edges_within_node) as u64;
                read_at(file, buf.as_mut_slice(), abs_offset)
                    .expect("read edges from on-disk node block");
                // SAFETY: buf is `cap * size_of::<NodeId>()` bytes; NodeId is repr(transparent)
                // over u32 so the underlying byte layout matches.
                unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const NodeId, cap) }
            }
        }
    }

    /// Copy the block's in-memory arena bytes to `path` without changing storage state.
    /// No-op (returns `Ok(())`) for on-disk or evicted blocks — they have no live arena
    /// to snapshot. The file format is identical to [`swap_out`]: raw arena bytes followed
    /// by a CRC32 checksum, so the output can be uploaded to S3 and later restored via
    /// [`swap_in_from`].
    pub fn copy_to(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let NodeBlockStorage::InMemory(arena) = &self.storage else {
            return Ok(());
        };
        let mapped = arena.mapped_bytes();
        // SAFETY: `arena.as_ptr()` is valid for `mapped` bytes (the anonymous mmap).
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(arena.as_ptr(), mapped) };
        let mut hasher = Crc32Hasher::new();
        hasher.update(bytes);
        let checksum = hasher.finalize();
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.write_all(bytes)?;
        file.write_all(&checksum.to_le_bytes())?;
        file.flush()
    }

    /// Write the block's in-memory arena bytes to `path` (truncating any existing file),
    /// release the mapping, and transition to [`NodeBlockStorage::OnDisk`]. Errors if the
    /// block is already on disk.
    pub fn swap_out(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let new_file = match &self.storage {
            NodeBlockStorage::InMemory(arena) => {
                let mapped = arena.mapped_bytes();
                // SAFETY: `arena.as_ptr()` is valid for `mapped` bytes (the anonymous mmap).
                let bytes: &[u8] = unsafe { std::slice::from_raw_parts(arena.as_ptr(), mapped) };
                let mut hasher = Crc32Hasher::new();
                hasher.update(bytes);
                let checksum = hasher.finalize();
                let mut file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)?;
                file.write_all(bytes)?;
                file.write_all(&checksum.to_le_bytes())?;
                file.flush()?;
                file
            }
            NodeBlockStorage::OnDisk(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "node block is already swapped out",
                ));
            }
            NodeBlockStorage::Evicted => {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "cannot swap_out an evicted node block; restore via swap_in_from first",
                ));
            }
        };
        // Assignment drops the previous storage (and the in-memory mmap inside it).
        self.storage = NodeBlockStorage::OnDisk(new_file);
        Ok(())
    }

    /// Drop the block's local backing (arena bytes or open fd) without touching any
    /// local file that may exist on disk. After this returns, [`is_evicted`] is `true`
    /// and read APIs (`vector_at`, `neighbors_at`) will panic until the block is
    /// restored via [`swap_in_from`].
    ///
    /// Use case: after `swap_out` followed by an upload to blob storage, call `evict`
    /// to close the fd so the caller can `std::fs::remove_file` the local copy and
    /// fully reclaim disk space — the block now lives only in remote storage.
    pub fn evict(&mut self) {
        self.storage = NodeBlockStorage::Evicted;
    }

    /// Open `path`, read its entire contents into a fresh anonymous arena, and transition
    /// to [`NodeBlockStorage::InMemory`]. Works from any current state — replaces the
    /// existing arena or closes the existing fd before opening `path`.
    ///
    /// Inverse of [`swap_out`] but accepts an arbitrary path, so the bytes can come from
    /// somewhere other than the original swap-out target (e.g. a fresh download from
    /// blob storage). The block's `len` (node count) and `block_index` are preserved.
    pub fn swap_in_from(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut file = OpenOptions::new().read(true).open(path.as_ref())?;
        let file_len = file.metadata()?.len() as usize;
        if file_len < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "arena file too small",
            ));
        }
        let arena_len = file_len - 4;
        let arena = Arena::try_with_capacity(DEFAULT_ARENA_CAPACITY)?;
        // SAFETY: anonymous mmap is at least `mapped_bytes() >= arena_len` bytes; we treat
        // the first `arena_len` bytes as the destination buffer for the file contents.
        let dest: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(arena.as_ptr() as *mut u8, arena.mapped_bytes())
        };
        file.read_exact(&mut dest[..arena_len])?;
        let mut crc_buf = [0u8; 4];
        file.read_exact(&mut crc_buf)?;
        let stored_crc = u32::from_le_bytes(crc_buf);
        let mut hasher = Crc32Hasher::new();
        hasher.update(&dest[..arena_len]);
        let computed_crc = hasher.finalize();
        if computed_crc != stored_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "arena CRC32 mismatch: expected {stored_crc:#010x}, got {computed_crc:#010x}"
                ),
            ));
        }
        // Seal the arena so future push_node calls return None.
        let cap = arena.capacity_bytes();
        let _ = arena.try_alloc_slice_aligned::<u8>(cap, 1);
        self.storage = NodeBlockStorage::InMemory(arena);
        Ok(())
    }

    /// Restore an [`NodeBlockStorage::OnDisk`] block to memory by reading the file into a
    /// fresh anonymous arena. Errors if the block is already in memory.
    ///
    /// The restored arena is sealed: bump = capacity, so further [`NodeBlock::push_node`]
    /// calls return `None`. Reads (`neighbors_at`, etc.) work normally.
    pub fn swap_in(&mut self) -> io::Result<()> {
        let new_arena = match &mut self.storage {
            NodeBlockStorage::OnDisk(file) => {
                let file_len = file.metadata()?.len() as usize;
                if file_len < 4 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "arena file too small",
                    ));
                }
                let arena_len = file_len - 4;
                let arena = Arena::try_with_capacity(DEFAULT_ARENA_CAPACITY)?;
                // SAFETY: anonymous mmap is at least `mapped_bytes() >= arena_len` bytes; we treat
                // the first `arena_len` bytes as the destination buffer for the file contents.
                let dest: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(arena.as_ptr() as *mut u8, arena.mapped_bytes())
                };
                file.seek(SeekFrom::Start(0))?;
                file.read_exact(&mut dest[..arena_len])?;
                let mut crc_buf = [0u8; 4];
                file.read_exact(&mut crc_buf)?;
                let stored_crc = u32::from_le_bytes(crc_buf);
                let mut hasher = Crc32Hasher::new();
                hasher.update(&dest[..arena_len]);
                let computed_crc = hasher.finalize();
                if computed_crc != stored_crc {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("arena CRC32 mismatch: expected {stored_crc:#010x}, got {computed_crc:#010x}"),
                    ));
                }
                // Seal: advance bump to the full capacity so future push_node calls return None.
                let cap = arena.capacity_bytes();
                let _ = arena.try_alloc_slice_aligned::<u8>(cap, 1);
                arena
            }
            NodeBlockStorage::InMemory(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "node block is already in memory",
                ));
            }
            NodeBlockStorage::Evicted => {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "node block is evicted; use swap_in_from(path) to restore from a file",
                ));
            }
        };
        self.storage = NodeBlockStorage::InMemory(new_arena);
        Ok(())
    }

    // get the neighbors of the node at the given level
    pub fn neighbors_at_mut(&mut self, node_id: NodeId, level: usize) -> &mut [NodeId] {
        assert!(level < LEVELS);
        let node_address = self.calculate_node_address(node_id);
        unsafe {
            Node::edges_at_level_mut(
                node_address as *mut u8,
                self.dim,
                level,
                self.m,
                self.m_max0,
            )
        }
    }

    pub fn save_neighbors(&mut self, node_id: NodeId, neighbors: &[NodeId], level: usize) {
        assert!(level < LEVELS);
        let node_address = self.calculate_node_address(node_id);
        let row = unsafe {
            Node::edges_at_level_mut(
                node_address as *mut u8,
                self.dim,
                level,
                self.m,
                self.m_max0,
            )
        };
        assert!(
            neighbors.len() <= row.len(),
            "neighbors len {} exceeds level capacity {}",
            neighbors.len(),
            row.len()
        );
        row[..neighbors.len()].copy_from_slice(neighbors);
    }

    // TODO: How to ensure the level is valid?
    pub fn ensure_level(&mut self, _: NodeId, _: usize) {}
}

pub struct ArenaNodeStore {
    dim: usize,
    m: usize,
    m_max0: usize,
    blocks: Vec<NodeBlock>,
}

impl ArenaNodeStore {
    pub fn try_new(dim: usize, m: usize, m_max0: usize) -> std::io::Result<Self> {
        Ok(Self {
            dim,
            m,
            m_max0,
            blocks: Vec::new(),
        })
    }

    /// Set `len` on each block by counting how many `node_ids` map to it.
    /// Called after [`load_from_dir`] + level loading so `len()` and `is_empty()`
    /// return correct values for a restored index.
    pub fn rebuild_lens_from_node_ids(&mut self, node_ids: &[NodeId]) {
        for block in &mut self.blocks {
            block.len = 0;
        }
        for &nid in node_ids {
            let bi = NodeBlock::derive_block_index(nid);
            if let Some(block) = self.blocks.get_mut(bi) {
                block.len += 1;
            }
        }
    }

    /// Populate this store from `block_*.arena` files in `dir`, creating one
    /// [`NodeBlock`] per file. Clears any existing blocks first. Used during crash
    /// recovery to reconstruct arena state from files uploaded to S3 by the snapshot
    /// task. Returns the number of blocks loaded.
    ///
    /// `len` on each restored block is 0 (node count is not serialized into the arena
    /// file); search correctness is unaffected because `vector_at`/`neighbors_at` use
    /// raw byte offsets, not the counter.
    pub fn load_from_dir(&mut self, dir: impl AsRef<Path>) -> io::Result<usize> {
        let dir = dir.as_ref();
        let mut entries: Vec<(usize, PathBuf)> = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("arena") {
                continue;
            }
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if let Some(idx_str) = stem.strip_prefix("block_") {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        entries.push((idx, path));
                    }
                }
            }
        }
        entries.sort_by_key(|(idx, _)| *idx);

        self.blocks.clear();
        for (block_index, path) in entries {
            // try_new allocates a fresh arena; swap_in_from immediately replaces it with
            // the on-disk bytes, so the initial mmap is dropped right away.
            let mut block = NodeBlock::try_new(self.dim, self.m, self.m_max0, block_index)
                .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "NodeBlock alloc failed"))?;
            block.swap_in_from(&path)?;
            self.blocks.push(block);
        }
        Ok(self.blocks.len())
    }

    /// Copy every in-memory block to `dir` as `block_<idx>.arena` without changing
    /// storage state. On-disk and evicted blocks are skipped. Returns the number of
    /// blocks written. The output format matches [`swap_out`] so files can be uploaded
    /// to S3 and later restored via [`swap_in_from`].
    pub fn snapshot_to_dir(&self, dir: impl AsRef<Path>) -> io::Result<usize> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let mut written = 0;
        for block in &self.blocks {
            if !block.is_in_memory() {
                continue;
            }
            let path = dir.join(format!("block_{}.arena", block.block_index));
            block.copy_to(&path)?;
            written += 1;
        }
        Ok(written)
    }

    /// Swap every in-memory block in this store to `dir`, one file per block named
    /// `block_<block_index>.arena`. The directory is created if missing.
    ///
    /// Already-on-disk blocks are skipped (not an error). Returns the number of blocks
    /// that transitioned from memory to disk in this call.
    pub fn swap_out(&mut self, dir: impl AsRef<Path>) -> io::Result<usize> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let mut moved = 0;
        for block in &mut self.blocks {
            if block.is_on_disk() {
                continue;
            }
            let path = dir.join(format!("block_{}.arena", block.block_index));
            block.swap_out(&path)?;
            moved += 1;
        }
        Ok(moved)
    }

    /// Swap every on-disk block back into memory. Already-in-memory blocks are skipped.
    /// Returns the number of blocks restored.
    pub fn swap_in(&mut self) -> io::Result<usize> {
        let mut restored = 0;
        for block in &mut self.blocks {
            if block.is_in_memory() {
                continue;
            }
            block.swap_in()?;
            restored += 1;
        }
        Ok(restored)
    }

    /// Drop the local backing (arena or fd) of every block. Already-evicted blocks
    /// are skipped. Returns the number of blocks that transitioned to evicted.
    ///
    /// Callers usually pair this with `std::fs::remove_dir_all(dir)` after a successful
    /// upload to free disk; the block can later be brought back with [`swap_in_from`].
    pub fn evict(&mut self) -> usize {
        let mut evicted = 0;
        for block in &mut self.blocks {
            if block.is_evicted() {
                continue;
            }
            block.evict();
            evicted += 1;
        }
        evicted
    }

    /// Restore every non-in-memory block by reading `dir/block_<block_index>.arena`.
    /// Files must exist for every evicted/on-disk block; missing files surface as I/O errors.
    /// Already-in-memory blocks are skipped. Returns the number of blocks restored.
    pub fn swap_in_from(&mut self, dir: impl AsRef<Path>) -> io::Result<usize> {
        let dir = dir.as_ref();
        let mut restored = 0;
        for block in &mut self.blocks {
            if block.is_in_memory() {
                continue;
            }
            let path = dir.join(format!("block_{}.arena", block.block_index));
            block.swap_in_from(&path)?;
            restored += 1;
        }
        Ok(restored)
    }

    /// `true` if every block is currently swapped to disk.
    pub fn all_on_disk(&self) -> bool {
        !self.blocks.is_empty() && self.blocks.iter().all(|b| b.is_on_disk())
    }

    /// `true` if every block is currently in memory.
    pub fn all_in_memory(&self) -> bool {
        self.blocks.iter().all(|b| b.is_in_memory())
    }

    /// `true` if every block is currently evicted (no local copy).
    pub fn all_evicted(&self) -> bool {
        !self.blocks.is_empty() && self.blocks.iter().all(|b| b.is_evicted())
    }

    #[inline]
    fn block(&self, block_index: usize) -> &NodeBlock {
        &self.blocks[block_index]
    }

    #[inline]
    fn block_mut(&mut self, block_index: usize) -> &mut NodeBlock {
        &mut self.blocks[block_index]
    }

    /// Removes `target` from `node_id`'s outgoing neighbors at `level` (sentinel layout).
    fn remove_outgoing_to(&mut self, node_id: NodeId, target: NodeId, level: usize) {
        if node_id == INVALID_NODE_ID || target == INVALID_NODE_ID {
            return;
        }
        let bi = NodeBlock::derive_block_index(node_id);
        if bi >= self.blocks.len() {
            return;
        }
        let block = self.block_mut(bi);
        let neighbors = block.neighbors_at_mut(node_id, level);
        let mut target_index = neighbors.len();
        let mut first_empty_index = neighbors.len();
        for i in 0..neighbors.len() {
            if neighbors[i] == target {
                target_index = i;
            }

            if neighbors[i] == INVALID_NODE_ID {
                first_empty_index = i;
                break;
            }
        }

        if target_index == neighbors.len() {
            return;
        }

        neighbors[target_index] = INVALID_NODE_ID;

        // this neighbor list is empty
        if first_empty_index == 0 {
            return;
        }
        // swap the last non empty neighbor to the evicted slot
        let last_non_empty_index = first_empty_index - 1;
        neighbors[target_index] = neighbors[last_non_empty_index];
        neighbors[last_non_empty_index] = INVALID_NODE_ID;
    }
}
/// Graph (node / edge) side of HNSW; see [`NaiveNodeStore`] and [`ArenaNodeStore`].
pub trait HnswNodeStore {
    fn len(&self) -> usize;
    /// Returns new internal id, or `None` if the store is at capacity.
    fn push_node(&mut self, vector: &[f32], max_level: usize) -> Option<NodeId>;
    /// `buf` is scratch space used only by the arena/on-disk path to stage edge bytes; the
    /// naive impl ignores it. Callers should reuse the same `Vec<u8>` across calls to avoid
    /// per-query allocations.
    fn neighbors_at<'a>(&'a self, id: NodeId, level: usize, buf: &'a mut Vec<u8>) -> &'a [NodeId];
    fn ensure_level(&mut self, id: NodeId, level: usize);
    // save the neighbors of a node to the edges at the given level
    fn save_neighbors(&mut self, id: NodeId, neighbors: &[NodeId], level: usize);
    /// `distance_fn` is a plain [`fn`] pointer so this trait stays **object-safe** (`dyn HnswNodeStore`,
    /// `Box<dyn HnswNodeStore>`). Use e.g. [`vector::distance::euclidean_distance_sq`].
    fn add_directed_edge(
        &mut self,
        src_id: NodeId,
        dst_id: NodeId,
        level: usize,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> bool;
    /// `buf` is scratch used only by the on-disk path; in-memory reads borrow from the
    /// arena and ignore it. Callers should reuse the same `Vec<u8>` across calls.
    fn vector_at<'a>(&'a self, id: NodeId, buf: &'a mut Vec<u8>) -> &'a [f32];

    /// Populate this store from `block_*.arena` files in `dir`, replacing any existing
    /// blocks. Used during crash recovery to reconstruct an arena from snapshot files
    /// downloaded from S3. Default: no-op (returns `Ok(0)`).
    fn load_from_dir(&mut self, _dir: &Path) -> io::Result<usize> {
        Ok(0)
    }
    /// Recompute block node-counts from a list of NodeIds. Called after
    /// [`load_from_dir`] + level loading during recovery. Default: no-op.
    fn rebuild_lens_from_node_ids(&mut self, _node_ids: &[NodeId]) {}
    /// Copy in-memory storage units to `dir` without changing storage state. Default:
    /// no-op (returns `Ok(0)`). [`ArenaNodeStore`] overrides this to fan out across its
    /// blocks. Output format matches [`Self::swap_out`] so files are S3-restorable.
    fn snapshot_to_dir(&self, _dir: &Path) -> io::Result<usize> {
        Ok(0)
    }
    /// Move underlying storage units to disk under `dir`. Default: no-op (return `Ok(0)`)
    /// for stores without arena-backed storage. [`ArenaNodeStore`] overrides this to
    /// fan out across its blocks.
    fn swap_out(&mut self, _dir: &Path) -> io::Result<usize> {
        Ok(0)
    }
    /// Restore on-disk storage units to memory. Default: no-op.
    fn swap_in(&mut self) -> io::Result<usize> {
        Ok(0)
    }
    /// Drop local backing (arena bytes or open fd) for every storage unit; bytes must
    /// be restored via [`Self::swap_in_from`] before the next read. Default: no-op.
    fn evict(&mut self) -> usize {
        0
    }
    /// Restore every storage unit by reading `dir/block_<i>.arena`. Default: no-op.
    fn swap_in_from(&mut self, _dir: &Path) -> io::Result<usize> {
        Ok(0)
    }
    /// Names of arena backing files produced by [`Self::swap_out`]. Default: empty.
    fn arena_file_names(&self) -> Vec<String> {
        Vec::new()
    }
}

impl HnswNodeStore for NaiveNodeStore {
    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn push_node(&mut self, vector: &[f32], max_level: usize) -> Option<NodeId> {
        let id = NodeId(self.nodes.len() as u32);
        self.vector_store.store_new(id, vector);
        self.nodes.push(GraphNode::new(max_level));
        Some(id)
    }

    fn neighbors_at<'a>(&'a self, id: NodeId, level: usize, _buf: &'a mut Vec<u8>) -> &'a [NodeId] {
        self.nodes[id.0 as usize].neighbors_at(level)
    }

    fn ensure_level(&mut self, id: NodeId, level: usize) {
        self.nodes[id.0 as usize].ensure_level(level);
    }

    fn save_neighbors(&mut self, id: NodeId, neighbors: &[NodeId], level: usize) {
        self.nodes[id.0 as usize].neighbors[level].extend(neighbors);
    }

    // update the edge from the node to the neighbor at the given level
    fn add_directed_edge(
        &mut self,
        src_id: NodeId,
        dst_id: NodeId,
        level: usize,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> bool {
        let cap = if level == 0 { self.m_max0 } else { self.m };
        if self.nodes[src_id.0 as usize].neighbors[level].contains(&dst_id) {
            return true;
        }
        if self.nodes[src_id.0 as usize].neighbors[level].len() < cap {
            self.nodes[src_id.0 as usize].neighbors[level].push(dst_id);
            return true;
        }

        let (farthest_neighbor_index, farthest_distance) = {
            let neighbors = &self.nodes[src_id.0 as usize].neighbors[level];
            let mut idx = 0usize;
            let mut dist = f32::MIN;
            let mut buf_a: Vec<u8> = Vec::new();
            let mut buf_b: Vec<u8> = Vec::new();
            for i in 0..neighbors.len() {
                let d = distance_fn(
                    self.vector_at(src_id, &mut buf_a),
                    self.vector_at(neighbors[i], &mut buf_b),
                );
                if d > dist {
                    idx = i;
                    dist = d;
                }
            }
            (idx, dist)
        };

        let mut buf_a: Vec<u8> = Vec::new();
        let mut buf_b: Vec<u8> = Vec::new();
        let new_distance = distance_fn(
            self.vector_at(src_id, &mut buf_a),
            self.vector_at(dst_id, &mut buf_b),
        );
        if new_distance > farthest_distance {
            return false;
        }

        let removed_neighbor =
            self.nodes[src_id.0 as usize].neighbors[level][farthest_neighbor_index];
        if removed_neighbor != src_id {
            self.nodes[removed_neighbor.0 as usize].neighbors[level].retain(|&x| x != src_id);
        }
        self.nodes[src_id.0 as usize].neighbors[level][farthest_neighbor_index] = dst_id;
        true
    }

    fn vector_at<'a>(&'a self, id: NodeId, _buf: &'a mut Vec<u8>) -> &'a [f32] {
        self.vector_store.vector_at(id)
    }
}

impl HnswNodeStore for ArenaNodeStore {
    fn len(&self) -> usize {
        self.blocks.iter().map(|b| b.len()).sum()
    }

    fn push_node(&mut self, vector: &[f32], max_level: usize) -> Option<NodeId> {
        if self.blocks.is_empty() {
            self.blocks
                .push(NodeBlock::try_new(self.dim, self.m, self.m_max0, 0).unwrap());
        }
        let mut is_new_block = false;
        loop {
            if let Some(block) = self.blocks.last_mut() {
                if let Some(node_id) = block.push_node(vector, max_level) {
                    return Some(node_id);
                }

                if is_new_block {
                    // already tried to allocate a new block. Return None to indicate that the store is at capacity
                    return None;
                }

                // failed to push the node to the last block. Try to allocate a new block
                let new_block =
                    NodeBlock::try_new(self.dim, self.m, self.m_max0, self.blocks.len()).unwrap();
                self.blocks.push(new_block);
                is_new_block = true;
            }
        }
    }

    fn neighbors_at<'a>(&'a self, id: NodeId, level: usize, buf: &'a mut Vec<u8>) -> &'a [NodeId] {
        let block_index = NodeBlock::derive_block_index(id);
        match self.blocks.get(block_index) {
            Some(block) => block.neighbors_at(id, level, buf),
            None => &[],
        }
    }

    fn ensure_level(&mut self, id: NodeId, level: usize) {
        let block_index = NodeBlock::derive_block_index(id);
        if let Some(block) = self.blocks.get_mut(block_index) {
            block.ensure_level(id, level);
        }
    }

    fn save_neighbors(&mut self, id: NodeId, neighbors: &[NodeId], level: usize) {
        let block_index = NodeBlock::derive_block_index(id);
        if let Some(block) = self.blocks.get_mut(block_index) {
            block.save_neighbors(id, neighbors, level);
        }
    }

    fn add_directed_edge(
        &mut self,
        src_id: NodeId,
        dst_id: NodeId,
        level: usize,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> bool {
        let src_block_index = NodeBlock::derive_block_index(src_id);
        if src_block_index >= self.blocks.len() {
            return false;
        }

        // {
        //     let block = self.block(src_block_index);
        //     let addr = block.calculate_node_address(src_id);
        //     if unsafe { level > Node::max_level(addr, block.dim) } {
        //         return false;
        //     }
        // }

        // Empty slot or duplicate — keep `src` borrow local.
        {
            let block = self.block_mut(src_block_index);
            let neighbors = block.neighbors_at_mut(src_id, level);
            for i in 0..neighbors.len() {
                if neighbors[i] == INVALID_NODE_ID {
                    neighbors[i] = dst_id;
                    return true;
                }
                if neighbors[i] == dst_id {
                    return true;
                }
            }
        }

        // Row full: pick farthest existing neighbor, then maybe swap. Snapshot neighbor ids first:
        // `neighbors_at_mut` borrows `self` mutably; `vector_at` needs `&self`.
        let neighbor_ids: Vec<NodeId> = {
            let block = self.block(src_block_index);
            let mut scratch: Vec<u8> = Vec::new();
            block.neighbors_at(src_id, level, &mut scratch).to_vec()
        };

        let (farthest_idx, farthest_dist, removed_neighbor, found) = {
            let mut farthest_neighbor_index = 0usize;
            let mut farthest_distance = f32::MIN;
            let mut found = false;
            let mut buf_a: Vec<u8> = Vec::new();
            let mut buf_b: Vec<u8> = Vec::new();
            for i in 0..neighbor_ids.len() {
                let nid = neighbor_ids[i];
                if nid == INVALID_NODE_ID {
                    continue;
                }
                let nb_i = NodeBlock::derive_block_index(nid);
                if nb_i >= self.blocks.len() {
                    continue;
                }
                let distance = distance_fn(
                    self.vector_at(src_id, &mut buf_a),
                    self.vector_at(nid, &mut buf_b),
                );
                if !found || distance >= farthest_distance {
                    farthest_neighbor_index = i;
                    farthest_distance = distance;
                    found = true;
                }
            }
            let removed = if found {
                neighbor_ids[farthest_neighbor_index]
            } else {
                INVALID_NODE_ID
            };
            (farthest_neighbor_index, farthest_distance, removed, found)
        };

        if !found {
            return false;
        }

        let mut buf_a: Vec<u8> = Vec::new();
        let mut buf_b: Vec<u8> = Vec::new();
        let new_distance = distance_fn(
            self.vector_at(src_id, &mut buf_a),
            self.vector_at(dst_id, &mut buf_b),
        );
        if new_distance > farthest_dist {
            return false;
        }

        if removed_neighbor != src_id {
            let rbi = NodeBlock::derive_block_index(removed_neighbor);
            if rbi < self.blocks.len() {
                self.remove_outgoing_to(removed_neighbor, src_id, level);
            }
        }

        let block = self.block_mut(src_block_index);
        let neighbors = block.neighbors_at_mut(src_id, level);
        neighbors[farthest_idx] = dst_id;
        true
    }

    fn vector_at<'a>(&'a self, id: NodeId, buf: &'a mut Vec<u8>) -> &'a [f32] {
        let block_index = NodeBlock::derive_block_index(id);
        let block = self.block(block_index);
        block.vector_at(id, buf)
    }

    fn load_from_dir(&mut self, dir: &Path) -> io::Result<usize> {
        ArenaNodeStore::load_from_dir(self, dir)
    }

    fn rebuild_lens_from_node_ids(&mut self, node_ids: &[NodeId]) {
        ArenaNodeStore::rebuild_lens_from_node_ids(self, node_ids);
    }

    fn snapshot_to_dir(&self, dir: &Path) -> io::Result<usize> {
        ArenaNodeStore::snapshot_to_dir(self, dir)
    }

    fn swap_out(&mut self, dir: &Path) -> io::Result<usize> {
        // Delegate to the inherent method on `ArenaNodeStore` (UFCS to disambiguate).
        ArenaNodeStore::swap_out(self, dir)
    }

    fn swap_in(&mut self) -> io::Result<usize> {
        ArenaNodeStore::swap_in(self)
    }

    fn evict(&mut self) -> usize {
        ArenaNodeStore::evict(self)
    }

    fn swap_in_from(&mut self, dir: &Path) -> io::Result<usize> {
        ArenaNodeStore::swap_in_from(self, dir)
    }

    fn arena_file_names(&self) -> Vec<String> {
        self.blocks
            .iter()
            .map(|b| format!("block_{}.arena", b.block_index))
            .collect()
    }
}
#[cfg(test)]
mod tests {
    //! Unit tests for [`NaiveNodeStore`], [`ArenaNodeStore`], and [`NodeBlock`] layout helpers.

    use super::*;
    use vector::distance::euclidean_distance_sq;

    /// [`NaiveNodeStore::push_node`] assigns contiguous [`NodeId`]s (0, 1, …), stores vectors, and
    /// allocates `max_level + 1` neighbor rows on each [`GraphNode`].
    #[test]
    fn naive_push_yields_sequential_ids_and_vectors() {
        let mut store = NaiveNodeStore::new(4, 8);
        assert_eq!(store.len(), 0);

        let a = store.push_node(&[1.0, 2.0, 3.0, 4.0], 1).expect("push");
        assert_eq!(a, NodeId(0));
        let b = store.push_node(&[5.0, 6.0, 7.0, 8.0], 0).expect("push");
        assert_eq!(b, NodeId(1));
        assert_eq!(store.len(), 2);

        let mut buf: Vec<u8> = Vec::new();
        assert_eq!(store.vector_at(NodeId(0), &mut buf), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(store.vector_at(NodeId(1), &mut buf), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(store.nodes[0].max_level(), 1);
        assert_eq!(store.nodes[1].max_level(), 0);
    }

    /// Naive `save_neighbors` appends to per-level `Vec`s (separate calls for levels 0 and 1).
    #[test]
    fn naive_save_neighbors_extends_level_lists() {
        let mut store = NaiveNodeStore::new(4, 8);
        let id = store.push_node(&[0.0; 4], 2).expect("push");
        let n0 = vec![NodeId(10), NodeId(11)];
        let n1 = vec![NodeId(20)];
        store.save_neighbors(id, &n0, 0);
        store.save_neighbors(id, &n1, 1);
        let mut buf: Vec<u8> = Vec::new();
        assert_eq!(
            store.neighbors_at(id, 0, &mut buf),
            &[NodeId(10), NodeId(11)]
        );
        assert_eq!(store.neighbors_at(id, 1, &mut buf), &[NodeId(20)]);
        assert!(store.neighbors_at(id, 2, &mut buf).is_empty());
    }

    /// [`GraphNode::ensure_level`] grows the neighbor slot table when connecting at a higher level
    /// than the node was created with.
    #[test]
    fn naive_ensure_level_extends_neighbor_structure() {
        let mut store = NaiveNodeStore::new(4, 8);
        let id = store.push_node(&[0.0; 4], 0).expect("push");
        assert_eq!(store.nodes[id.0 as usize].neighbors.len(), 1);
        store.ensure_level(id, 3);
        assert_eq!(store.nodes[id.0 as usize].neighbors.len(), 4);
        assert_eq!(store.nodes[id.0 as usize].max_level(), 3);
    }

    /// Strips [`INVALID_NODE_ID`] sentinels from a fixed-width arena neighbor row for assertions.
    fn nonzero_neighbors(slice: &[NodeId]) -> Vec<NodeId> {
        slice
            .iter()
            .copied()
            .filter(|&x| x != INVALID_NODE_ID)
            .collect()
    }

    /// Shared geometry for eviction / back-edge tests: four 2-D points on the x-axis.
    const COLLINEAR_DIM: usize = 2;
    const COLLINEAR_M: usize = 2;
    const COLLINEAR_M_MAX0: usize = 2;
    /// From origin, distances² are 1, 10_000, 40_000 — third outgoing replaces `[200,0]` with `[1,0]`.
    const COLLINEAR_MAX_LEVEL: usize = 1;

    fn collinear_four_vectors() -> [[f32; 2]; 4] {
        [[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [1.0, 0.0]]
    }

    /// First `add_directed_edge` inserts; a second identical edge is a no-op but still reports success.
    /// Naive uses contiguous ids; arena uses sentinel-filled rows and encoded ids.
    #[test]
    fn add_directed_edge_inserts_once_and_duplicate_ok_naive_and_arena() {
        let test_cases: Vec<Box<dyn HnswNodeStore + 'static>> = vec![
            Box::new(NaiveNodeStore::new(4, 8)),
            Box::new(ArenaNodeStore::try_new(2, 4, 8).expect("new")),
        ];
        for mut store in test_cases {
            let id0 = store.push_node(&[0.0, 0.0], 1).expect("n0");
            let id1 = store.push_node(&[1.0, 0.0], 1).expect("n1");
            assert!(store.add_directed_edge(id0, id1, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id0, id1, 0, euclidean_distance_sq));
            let mut buf: Vec<u8> = Vec::new();
            assert_eq!(
                nonzero_neighbors(store.neighbors_at(id0, 0, &mut buf)),
                vec![id1]
            );
        }
    }

    /// With level-0 capacity two, a third outgoing edge evicts the farthest neighbor (same graph on
    /// both stores; arena uses encoded [`NodeId`]s and sentinels).
    #[test]
    fn add_directed_edge_evicts_farthest_when_level_zero_full_naive_and_arena() {
        let test_cases: Vec<Box<dyn HnswNodeStore + 'static>> = vec![
            Box::new(NaiveNodeStore::new(COLLINEAR_M, COLLINEAR_M_MAX0)),
            Box::new(
                ArenaNodeStore::try_new(COLLINEAR_DIM, COLLINEAR_M, COLLINEAR_M_MAX0).expect("new"),
            ),
        ];
        let v = collinear_four_vectors();
        for mut store in test_cases {
            let id0 = store.push_node(&v[0], COLLINEAR_MAX_LEVEL).expect("n0");
            let id1 = store.push_node(&v[1], COLLINEAR_MAX_LEVEL).expect("n1");
            let id2 = store.push_node(&v[2], COLLINEAR_MAX_LEVEL).expect("n2");
            let id3 = store.push_node(&v[3], COLLINEAR_MAX_LEVEL).expect("n3");
            assert!(store.add_directed_edge(id0, id1, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id0, id2, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id0, id3, 0, euclidean_distance_sq));
            let mut buf: Vec<u8> = Vec::new();
            let present = nonzero_neighbors(store.neighbors_at(id0, 0, &mut buf));
            assert_eq!(present.len(), 2);
            assert!(present.contains(&id1));
            assert!(present.contains(&id3));
            assert!(!present.contains(&id2));
        }
    }

    /// Full row eviction removes the reverse edge from the dropped neighbor (naive `Vec` vs arena
    /// sentinels + `remove_outgoing_to`).
    #[test]
    fn add_directed_edge_drops_back_edge_from_evicted_neighbor_naive_and_arena() {
        let test_cases: Vec<Box<dyn HnswNodeStore + 'static>> = vec![
            Box::new(NaiveNodeStore::new(COLLINEAR_M, COLLINEAR_M_MAX0)),
            Box::new(
                ArenaNodeStore::try_new(COLLINEAR_DIM, COLLINEAR_M, COLLINEAR_M_MAX0).expect("new"),
            ),
        ];
        let v = collinear_four_vectors();
        for mut store in test_cases {
            let id0 = store.push_node(&v[0], COLLINEAR_MAX_LEVEL).expect("n0");
            let id1 = store.push_node(&v[1], COLLINEAR_MAX_LEVEL).expect("n1");
            let id2 = store.push_node(&v[2], COLLINEAR_MAX_LEVEL).expect("n2");
            let id3 = store.push_node(&v[3], COLLINEAR_MAX_LEVEL).expect("n3");
            assert!(store.add_directed_edge(id0, id1, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id0, id2, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id1, id0, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id2, id0, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id0, id3, 0, euclidean_distance_sq));
            let mut buf: Vec<u8> = Vec::new();
            assert!(!nonzero_neighbors(store.neighbors_at(id2, 0, &mut buf)).contains(&id0));
            assert!(nonzero_neighbors(store.neighbors_at(id1, 0, &mut buf)).contains(&id0));
        }
    }

    /// Cannot attach edges at a level higher than the node’s allocated neighbor rows.
    #[test]
    fn arena_add_directed_edge_returns_false_for_level_above_max_level() {
        let mut store = ArenaNodeStore::try_new(2, 2, 2).expect("new");
        let id0 = store.push_node(&[0.0, 0.0], 0).expect("push");
        let id1 = store.push_node(&[1.0, 0.0], 0).expect("push");
        assert!(!store.add_directed_edge(id0, id1, 1, euclidean_distance_sq));
    }

    /// Single [`NodeBlock`]: allocation alignment, vector/`max_level` fields, packed edges,
    /// `save_neighbors`, and raw `Node::*` slice views per level.
    #[test]
    fn node_data_store() {
        const DIM: usize = 128;
        const M: usize = 16;
        const M_MAX0: usize = 32;
        let mut node_block = NodeBlock::try_new(DIM, M, M_MAX0, 0).expect("test alloc");
        for i in 0..10 {
            let fill = 1.0f32 * i as f32;
            let stored = vec![fill; DIM];
            let max_level = ((i + 5) * 11usize).pow(5).min(MAX_LEVEL);
            let node_id = node_block
                .push_node(&stored, max_level)
                .expect("test alloc");
            let node_address = node_block.calculate_node_address(node_id);
            // vector is aligned to 8 bytes
            assert_eq!(node_address as usize % 8, 0);

            assert_eq!(
                unsafe { Node::vector(node_address as *mut u8, DIM) },
                stored.as_slice()
            );

            let edge_slots = Node::edge_count(max_level, M, M_MAX0);
            let expected_edges = vec![INVALID_NODE_ID; edge_slots];
            assert_eq!(
                unsafe { Node::edges(node_address as *mut u8, DIM, max_level, M, M_MAX0) },
                expected_edges.as_slice()
            );

            // save some edges
            for l in 0..max_level + 1 {
                let num_neighbors = if l == 0 { M_MAX0 } else { M };
                let neighbors = vec![NodeId((i * (max_level + 10) + l) as u32); num_neighbors];
                node_block.save_neighbors(node_id, &neighbors.as_slice(), l);
            }

            // validate the edges
            for l in 0..max_level + 1 {
                let num_neighbors = if l == 0 { M_MAX0 } else { M };
                let expected_neighbors =
                    vec![NodeId((i * (max_level + 10) + l) as u32); num_neighbors];
                let mut buf: Vec<u8> = Vec::new();
                let actual_neighbors = node_block.neighbors_at(node_id, l, &mut buf);
                assert_eq!(actual_neighbors.len(), num_neighbors);
                assert_eq!(actual_neighbors, expected_neighbors.as_slice(), "neighbors at i {i} node {node_id:?} at level {l} should be {expected_neighbors:?}");

                // test neighbors_at_mut
                let neighbors_at_mut = node_block.neighbors_at_mut(node_id, l);
                assert_eq!(neighbors_at_mut.len(), num_neighbors);
                assert_eq!(neighbors_at_mut, expected_neighbors.as_slice(), "neighbors at i {i} node {node_id:?} at level {l} should be {expected_neighbors:?}");

                // test using unsafe api
                let unsafe_neighbors =
                    unsafe { Node::edges_at_level(node_address as *mut u8, DIM, l, M, M_MAX0) };
                assert_eq!(unsafe_neighbors.len(), num_neighbors);
                assert_eq!(unsafe_neighbors, expected_neighbors.as_slice(), "neighbors at i {i} node {node_id:?} at level {l} should be {expected_neighbors:?}");

                // test using unsafe api with mut
                let unsafe_neighbors_mut =
                    unsafe { Node::edges_at_level_mut(node_address as *mut u8, DIM, l, M, M_MAX0) };
                assert_eq!(unsafe_neighbors_mut.len(), num_neighbors);
                assert_eq!(unsafe_neighbors_mut, expected_neighbors.as_slice(), "neighbors at i {i} node {node_id:?} at level {l} should be {expected_neighbors:?}");

                // test the unsafe edges api
                let unsafe_edges =
                    unsafe { Node::edges(node_address as *mut u8, DIM, max_level, M, M_MAX0) };
                assert_eq!(unsafe_edges.len(), edge_slots);
                // extract the edges for this level and compare with the expected neighbors
                let start_index = if l == 0 { 0 } else { M_MAX0 + (l - 1) * M };
                let end_index = if l == max_level {
                    edge_slots
                } else {
                    M_MAX0 + l * M
                };
                let edges = unsafe_edges[start_index..end_index].to_vec();
                assert_eq!(
                    edges,
                    expected_neighbors.as_slice(),
                    "edges at i {i} node {node_id:?} at level {l} should be {expected_neighbors:?}"
                );
            }
        }
    }

    /// [`ArenaNodeStore`] spanning multiple [`NodeBlock`]s: push until several blocks exist, then
    /// verify `vector_at`, `max_level`, and filled neighbor rows round-trip per encoded id.
    #[test]
    fn multiple_arena_stores() {
        const DIM: usize = 128;
        const M: usize = 16;
        const M_MAX0: usize = 32;
        let mut store = ArenaNodeStore::try_new(DIM, M, M_MAX0).expect("new");
        let max_num_blocks = 4;
        let mut i = 0;
        let mut node_ids: Vec<NodeId> = Vec::new();
        while store.blocks.len() < max_num_blocks {
            let fill = 1.0f32 * i as f32;
            let stored = vec![fill; DIM];
            let max_level = ((i + 6) * 11usize).pow(2).min(MAX_LEVEL);

            let node_id = store.push_node(&stored, max_level).expect("test alloc");

            node_ids.push(node_id);

            // initialize the neighbors
            for l in 0..max_level + 1 {
                let num_neighbors = if l == 0 { M_MAX0 } else { M };
                let neighbors = vec![NodeId((i * (max_level + 10) + l) as u32); num_neighbors];
                store.save_neighbors(node_id, &neighbors.as_slice(), l);
            }
            i += 1;
        }

        for i in 0..node_ids.len() {
            let node_id = node_ids[i];
            let block_index = NodeBlock::derive_block_index(node_id);
            let block = store.block(block_index);
            let node_address = block.calculate_node_address(node_id);
            assert_eq!(node_address as usize % 8, 0);

            let fill = 1.0f32 * i as f32;
            let stored = vec![fill; DIM];
            let max_level = ((i + 6) * 11usize).pow(2).min(MAX_LEVEL);

            // check data by arena storage api
            let mut buf: Vec<u8> = Vec::new();
            assert_eq!(
                store.vector_at(node_id, &mut buf),
                stored.as_slice(),
                "vector at node {node_id:?} should be {stored:?}"
            );

            // validate the neighbors
            for l in 0..max_level + 1 {
                let num_neighbors = if l == 0 { M_MAX0 } else { M };
                let expected_neighbors =
                    vec![NodeId((i * (max_level + 10) + l) as u32); num_neighbors];
                let mut buf: Vec<u8> = Vec::new();
                let actual_neighbors = store.neighbors_at(node_id, l, &mut buf);
                assert_eq!(actual_neighbors.len(), num_neighbors);
                assert_eq!(actual_neighbors, expected_neighbors.as_slice(), "neighbors at i {i} node {node_id:?} at level {l} should be {expected_neighbors:?}");
            }
        }
    }

    /// Unique temp dir for swap tests; caller responsible for cleanup.
    fn unique_swap_dir(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let pid = std::process::id();
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("mem_weaver_nodeblock_{tag}_{pid}_{nanos}_{n}"))
    }

    /// Deletes the directory tree when dropped.
    struct DirGuard(std::path::PathBuf);
    impl Drop for DirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn node_block_swap_round_trip_preserves_neighbors() {
        const DIM: usize = 8;
        const M: usize = 4;
        const M_MAX0: usize = 8;
        let dir = unique_swap_dir("rt");
        let _guard = DirGuard(dir.clone());
        std::fs::create_dir_all(&dir).expect("mk dir");

        let mut block = NodeBlock::try_new(DIM, M, M_MAX0, 0).expect("alloc block");
        // Insert two nodes and write neighbors at level 0 + level 1.
        let v0 = vec![1.0_f32; DIM];
        let v1 = vec![2.0_f32; DIM];
        let id0 = block.push_node(&v0, 1).expect("push v0");
        let id1 = block.push_node(&v1, 1).expect("push v1");
        block.save_neighbors(id0, &[id1], 0);
        block.save_neighbors(id1, &[id0], 0);
        block.save_neighbors(id0, &[id1], 1);

        // Capture the in-memory neighbors as ground truth before swap.
        let mut buf: Vec<u8> = Vec::new();
        let in_mem_id0_l0 = block.neighbors_at(id0, 0, &mut buf).to_vec();
        let in_mem_id1_l0 = block.neighbors_at(id1, 0, &mut buf).to_vec();
        let in_mem_id0_l1 = block.neighbors_at(id0, 1, &mut buf).to_vec();

        let path = dir.join("block_0.arena");
        block.swap_out(&path).expect("swap_out");
        assert!(block.is_on_disk());
        assert!(!block.is_in_memory());
        assert!(path.exists());

        // Reads must work while swapped out — this is the on-disk `read_at` path.
        let mut buf: Vec<u8> = Vec::new();
        assert_eq!(
            block.neighbors_at(id0, 0, &mut buf),
            in_mem_id0_l0.as_slice()
        );
        assert_eq!(
            block.neighbors_at(id1, 0, &mut buf),
            in_mem_id1_l0.as_slice()
        );
        assert_eq!(
            block.neighbors_at(id0, 1, &mut buf),
            in_mem_id0_l1.as_slice()
        );
        // And the convenience nonzero_neighbors filter agrees on the on-disk read.
        assert_eq!(
            nonzero_neighbors(block.neighbors_at(id0, 0, &mut buf)),
            vec![id1]
        );
        assert_eq!(
            nonzero_neighbors(block.neighbors_at(id1, 0, &mut buf)),
            vec![id0]
        );
        assert_eq!(
            nonzero_neighbors(block.neighbors_at(id0, 1, &mut buf)),
            vec![id1]
        );

        block.swap_in().expect("swap_in");
        assert!(block.is_in_memory());

        // Reads after swap_in observe the same neighbor rows.
        assert_eq!(
            block.neighbors_at(id0, 0, &mut buf),
            in_mem_id0_l0.as_slice()
        );
        assert_eq!(
            block.neighbors_at(id1, 0, &mut buf),
            in_mem_id1_l0.as_slice()
        );
        assert_eq!(
            block.neighbors_at(id0, 1, &mut buf),
            in_mem_id0_l1.as_slice()
        );

        // Sealed contract: no further pushes after a round-trip.
        let v2 = vec![3.0_f32; DIM];
        assert!(
            block.push_node(&v2, 0).is_none(),
            "swapped-in block must be sealed"
        );
    }

    #[test]
    fn node_block_neighbors_at_reads_from_disk_when_swapped_out() {
        // Multi-node, multi-level block: verifies the on-disk offset math
        // (node_offset + edges_header + per-level offset) for several (id, level) combinations.
        const DIM: usize = 16;
        const M: usize = 4;
        const M_MAX0: usize = 8;
        const N: usize = 5;
        const MAX_L: usize = 2;
        let dir = unique_swap_dir("read_from_disk");
        let _guard = DirGuard(dir.clone());
        std::fs::create_dir_all(&dir).expect("mk dir");

        let mut block =
            NodeBlock::try_new(DIM, M, M_MAX0, 7 /* nonzero block_index */).expect("alloc block");
        let mut ids: Vec<NodeId> = Vec::with_capacity(N);
        for i in 0..N {
            let v: Vec<f32> = (0..DIM).map(|j| (i * DIM + j) as f32).collect();
            ids.push(block.push_node(&v, MAX_L).expect("push"));
        }
        // Distinctive neighbor rows per (id, level) so an offset bug would mis-route them.
        for (i, &id) in ids.iter().enumerate() {
            for l in 0..=MAX_L {
                let cap = if l == 0 { M_MAX0 } else { M };
                let neighbors: Vec<NodeId> = (0..cap)
                    .map(|j| NodeId(((i + 1) * 1000 + l * 100 + j) as u32))
                    .collect();
                block.save_neighbors(id, &neighbors, l);
            }
        }

        // Snapshot every (id, level) row while in memory.
        let mut buf: Vec<u8> = Vec::new();
        let mut snapshots: Vec<(NodeId, usize, Vec<NodeId>)> = Vec::new();
        for &id in &ids {
            for l in 0..=MAX_L {
                let row = block.neighbors_at(id, l, &mut buf).to_vec();
                snapshots.push((id, l, row));
            }
        }

        block.swap_out(&dir.join("block.arena")).expect("swap_out");
        assert!(block.is_on_disk());

        // Read everything back while the block is still on disk; must match the snapshot.
        for (id, l, expected) in &snapshots {
            let got = block.neighbors_at(*id, *l, &mut buf);
            assert_eq!(
                got,
                expected.as_slice(),
                "on-disk read mismatch at id={id:?} level={l}"
            );
        }
    }

    #[test]
    fn node_block_swap_out_on_already_on_disk_errors() {
        let dir = unique_swap_dir("double_out");
        let _guard = DirGuard(dir.clone());
        std::fs::create_dir_all(&dir).expect("mk dir");

        let mut block = NodeBlock::try_new(4, 4, 8, 0).expect("alloc");
        let _ = block.push_node(&[0.0_f32; 4], 0).expect("push");
        let p = dir.join("block_0.arena");
        block.swap_out(&p).expect("first out");
        let err = block.swap_out(&p).expect_err("second swap_out must error");
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert!(block.is_on_disk(), "state unchanged on error");
    }

    #[test]
    fn node_block_swap_in_on_in_memory_errors() {
        let mut block = NodeBlock::try_new(4, 4, 8, 0).expect("alloc");
        let err = block
            .swap_in()
            .expect_err("swap_in on hot block must error");
        assert_eq!(err.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn arena_node_store_swap_fans_out_over_blocks() {
        const DIM: usize = 4;
        let dir = unique_swap_dir("store_fanout");
        let _guard = DirGuard(dir.clone());

        let mut store = ArenaNodeStore::try_new(DIM, 4, 8).expect("new store");
        // Force at least 2 blocks by pushing many nodes — but DEFAULT_ARENA_CAPACITY is large,
        // so just push 3 and verify swap is idempotent across all of them.
        for i in 0..3 {
            let v = vec![i as f32; DIM];
            store.push_node(&v, 0).expect("push");
        }
        assert!(store.all_in_memory());

        let moved = store.swap_out(&dir).expect("swap_out store");
        assert!(moved >= 1, "at least one block must have been swapped out");
        assert!(store.all_on_disk());

        // Idempotent: swap_out on already-on-disk store moves 0 blocks (no error).
        let moved_again = store.swap_out(&dir).expect("idempotent swap_out");
        assert_eq!(moved_again, 0);

        let restored = store.swap_in().expect("swap_in store");
        assert_eq!(
            restored, moved,
            "every previously-cold block must be restored"
        );
        assert!(store.all_in_memory());
    }

    // ── copy_to / snapshot_to_dir ─────────────────────────────────────────────

    /// `copy_to` writes the same bytes as `swap_out`, leaving the block in-memory.
    #[test]
    fn node_block_copy_to_matches_swap_out_bytes_and_preserves_state() {
        const DIM: usize = 8;
        const M: usize = 4;
        const M_MAX0: usize = 8;
        let dir = unique_swap_dir("copy_to_match");
        let _guard = DirGuard(dir.clone());
        std::fs::create_dir_all(&dir).expect("mkdir");

        // Insert a few nodes into two separate blocks so we have real data.
        let mut block_a = NodeBlock::try_new(DIM, M, M_MAX0, 0).expect("alloc");
        let mut block_b = NodeBlock::try_new(DIM, M, M_MAX0, 1).expect("alloc");
        for i in 0..4u32 {
            let v: Vec<f32> = (0..DIM)
                .map(|j| (i * DIM as u32 + j as u32) as f32)
                .collect();
            block_a.push_node(&v, 1).expect("push a");
            block_b.push_node(&v, 0).expect("push b");
        }

        // copy_to must not change state.
        block_a
            .copy_to(dir.join("copy_a.arena"))
            .expect("copy_to a");
        block_b
            .copy_to(dir.join("copy_b.arena"))
            .expect("copy_to b");
        assert!(
            block_a.is_in_memory(),
            "block_a must stay in-memory after copy_to"
        );
        assert!(
            block_b.is_in_memory(),
            "block_b must stay in-memory after copy_to"
        );

        // swap_out on fresh blocks produces the reference bytes.
        let mut ref_a = NodeBlock::try_new(DIM, M, M_MAX0, 0).expect("alloc ref_a");
        let mut ref_b = NodeBlock::try_new(DIM, M, M_MAX0, 1).expect("alloc ref_b");
        for i in 0..4u32 {
            let v: Vec<f32> = (0..DIM)
                .map(|j| (i * DIM as u32 + j as u32) as f32)
                .collect();
            ref_a.push_node(&v, 1).expect("push ref_a");
            ref_b.push_node(&v, 0).expect("push ref_b");
        }
        ref_a
            .swap_out(dir.join("swap_a.arena"))
            .expect("swap_out ref_a");
        ref_b
            .swap_out(dir.join("swap_b.arena"))
            .expect("swap_out ref_b");

        assert_eq!(
            std::fs::read(dir.join("copy_a.arena")).unwrap(),
            std::fs::read(dir.join("swap_a.arena")).unwrap(),
            "copy_to and swap_out must produce identical bytes for block_a"
        );
        assert_eq!(
            std::fs::read(dir.join("copy_b.arena")).unwrap(),
            std::fs::read(dir.join("swap_b.arena")).unwrap(),
            "copy_to and swap_out must produce identical bytes for block_b"
        );
    }

    /// `copy_to` on an on-disk block is a no-op (returns Ok without touching the file).
    #[test]
    fn node_block_copy_to_on_disk_is_noop() {
        let dir = unique_swap_dir("copy_to_ondisk");
        let _guard = DirGuard(dir.clone());
        std::fs::create_dir_all(&dir).expect("mkdir");

        let mut block = NodeBlock::try_new(4, 4, 8, 0).expect("alloc");
        block.push_node(&[1.0_f32; 4], 0).expect("push");
        block.swap_out(dir.join("block_0.arena")).expect("swap_out");
        assert!(block.is_on_disk());

        // copy_to on an on-disk block must succeed silently and not write a new file.
        block
            .copy_to(dir.join("ghost.arena"))
            .expect("copy_to on-disk");
        assert!(
            !dir.join("ghost.arena").exists(),
            "no file written for on-disk block"
        );
    }

    /// `copy_to` on an evicted block is a no-op.
    #[test]
    fn node_block_copy_to_evicted_is_noop() {
        let dir = unique_swap_dir("copy_to_evicted");
        let _guard = DirGuard(dir.clone());
        std::fs::create_dir_all(&dir).expect("mkdir");

        let mut block = NodeBlock::try_new(4, 4, 8, 0).expect("alloc");
        block.push_node(&[2.0_f32; 4], 0).expect("push");
        block.swap_out(dir.join("block_0.arena")).expect("swap_out");
        block.evict();
        assert!(block.is_evicted());

        block
            .copy_to(dir.join("ghost.arena"))
            .expect("copy_to evicted");
        assert!(
            !dir.join("ghost.arena").exists(),
            "no file written for evicted block"
        );
    }

    /// `snapshot_to_dir` writes files byte-identical to `swap_out` without changing state.
    #[test]
    fn arena_snapshot_to_dir_matches_swap_out_and_preserves_state() {
        const DIM: usize = 4;
        let snap_dir = unique_swap_dir("snap_match");
        let swap_dir = unique_swap_dir("swap_match");
        let _g1 = DirGuard(snap_dir.clone());
        let _g2 = DirGuard(swap_dir.clone());

        let mut store = ArenaNodeStore::try_new(DIM, 4, 8).expect("new");
        for i in 0..3u32 {
            store.push_node(&[i as f32; DIM], 0).expect("push");
        }
        assert!(store.all_in_memory());

        let written = store.snapshot_to_dir(&snap_dir).expect("snapshot_to_dir");
        assert!(written >= 1, "at least one block written");
        assert!(
            store.all_in_memory(),
            "store must remain in-memory after snapshot"
        );

        // Reference: swap an identical store to disk.
        let mut ref_store = ArenaNodeStore::try_new(DIM, 4, 8).expect("ref store");
        for i in 0..3u32 {
            ref_store.push_node(&[i as f32; DIM], 0).expect("push ref");
        }
        let moved = ref_store.swap_out(&swap_dir).expect("swap_out ref");
        assert_eq!(
            written, moved,
            "snapshot and swap must process the same number of blocks"
        );

        // Every arena file produced by snapshot must match the corresponding swap file.
        for entry in std::fs::read_dir(&snap_dir).expect("read snap_dir") {
            let path = entry.expect("entry").path();
            if path.extension().and_then(|s| s.to_str()) != Some("arena") {
                continue;
            }
            let filename = path.file_name().unwrap();
            let snap_bytes = std::fs::read(&path).expect("read snap file");
            let swap_bytes = std::fs::read(swap_dir.join(filename)).expect("read swap file");
            assert_eq!(
                snap_bytes, swap_bytes,
                "{filename:?}: snapshot bytes differ from swap_out bytes"
            );
        }
    }

    /// `snapshot_to_dir` on a store with no in-memory blocks writes nothing.
    #[test]
    fn arena_snapshot_to_dir_skips_on_disk_blocks() {
        const DIM: usize = 4;
        let swap_dir = unique_swap_dir("snap_skip_swap");
        let snap_dir = unique_swap_dir("snap_skip_snap");
        let _g1 = DirGuard(swap_dir.clone());
        let _g2 = DirGuard(snap_dir.clone());

        let mut store = ArenaNodeStore::try_new(DIM, 4, 8).expect("new");
        store.push_node(&[1.0_f32; DIM], 0).expect("push");
        store.swap_out(&swap_dir).expect("swap_out");
        assert!(store.all_on_disk());

        let written = store.snapshot_to_dir(&snap_dir).expect("snapshot_to_dir");
        assert_eq!(written, 0, "no in-memory blocks means nothing to snapshot");
        assert!(
            std::fs::read_dir(&snap_dir)
                .expect("read_dir")
                .next()
                .is_none(),
            "no files written when all blocks are on disk"
        );
    }

    /// Files produced by `snapshot_to_dir` pass CRC32 validation via `swap_in_from`.
    #[test]
    fn arena_snapshot_to_dir_files_are_restorable() {
        const DIM: usize = 4;
        const M: usize = 4;
        const M_MAX0: usize = 8;
        let snap_dir = unique_swap_dir("snap_restore");
        let _guard = DirGuard(snap_dir.clone());

        let mut store = ArenaNodeStore::try_new(DIM, M, M_MAX0).expect("new");
        for i in 0..4u32 {
            store.push_node(&[i as f32; DIM], 0).expect("push");
        }
        store.snapshot_to_dir(&snap_dir).expect("snapshot_to_dir");

        // Evict so swap_in_from is needed to restore.
        store
            .swap_out(&snap_dir)
            .expect("swap_out (transition to on-disk)");
        store.evict();
        assert!(store.all_evicted());

        let restored = store
            .swap_in_from(&snap_dir)
            .expect("swap_in_from snapshot files");
        assert!(restored >= 1, "at least one block must have been restored");
        assert!(store.all_in_memory());

        // The fact that swap_in_from succeeded with valid CRC32 proves the bytes
        // round-tripped correctly. Len check provides an additional sanity signal.
        assert_eq!(store.len(), 4);
    }
}
