use common::types::{InternalId, VectorId};
use mem::Arena;
use std::fmt;
/// Multi-field or vector block could not be allocated in the arena (capacity exhausted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorAllocFailed;

impl fmt::Display for VectorAllocFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("vector allocation failed: arena capacity exhausted")
    }
}

impl std::error::Error for VectorAllocFailed {}

pub struct Vector {}

impl Vector {
    #[inline]
    pub fn calculate_internal_id(
        block_index: usize,
        vector_index: usize,
        num_vectors_per_block: usize,
    ) -> InternalId {
        InternalId((block_index * num_vectors_per_block + vector_index) as u32)
    }

    #[inline]
    pub fn derive_block_index(internal_id: InternalId, num_vectors_per_block: usize) -> usize {
        internal_id.0 as usize / num_vectors_per_block
    }

    #[inline]
    pub fn derive_vector_index(internal_id: InternalId, num_vectors_per_block: usize) -> usize {
        internal_id.0 as usize % num_vectors_per_block
    }
}
/// A flat, arena-backed collection of fixed-dimension vectors for a single
/// named field. Float data is 32-byte aligned for SIMD; id slots live in a
/// separate arena region (`capacity` × [`VectorId`]).
///
/// Memory layout:
///   data: [ v0[0..D] | v1[0..D] | ... ]
///   ids:  [ id0 | id1 | ... ]
///
pub struct VectorBlock {
    // Holds the mmap backing `ids_ptr` / `data_ptr`; must outlive raw pointers.
    #[allow(dead_code)]
    arena: Arena,
    dim: usize,
    capacity: usize,
    ids_ptr: *mut VectorId,
    data_ptr: *mut f32,
    len: usize,
}

impl VectorBlock {
    pub fn try_new(dim: usize, arena_capacity: usize) -> Option<Self> {
        if arena_capacity == 0 || dim == 0 {
            return None;
        }

        let arena = Arena::try_with_capacity(arena_capacity).ok()?;
        let num_vectors = VectorBlock::max_vectors(arena.capacity_bytes(), dim);
        let data_ptr = {
            let slice = arena.try_alloc_vector_aligned(dim * num_vectors, 64)?;
            slice.fill(0.0);
            slice.as_mut_ptr()
        };
        let ids_ptr = {
            let ids = arena.try_alloc_slice_zeroed::<VectorId>(num_vectors)?;
            ids.as_mut_ptr()
        };

        Some(Self {
            arena,
            dim,
            capacity: num_vectors,
            ids_ptr,
            data_ptr,
            len: 0,
        })
    }

    /// Append a vector. Returns its position index, or `None` if full.
    pub fn push(&mut self, id: VectorId, data: &[f32]) -> Option<usize> {
        assert_eq!(data.len(), self.dim, "dimension mismatch on push");
        if self.len >= self.capacity {
            return None;
        }
        let idx = self.len;
        // SAFETY: idx < capacity, allocation covers full range.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.data_ptr.add(idx * self.dim),
                self.dim,
            );
            self.ids_ptr.add(idx).write(id);
        }
        self.len += 1;
        Some(idx)
    }

    /// Get vector slice and its id by position index.
    pub fn get(&self, idx: usize) -> Option<(VectorId, &[f32])> {
        if idx >= self.len {
            return None;
        }
        // SAFETY: idx < len, pointer valid.
        let slice =
            unsafe { std::slice::from_raw_parts(self.data_ptr.add(idx * self.dim), self.dim) };
        let id = unsafe { *self.ids_ptr.add(idx) };
        Some((id, slice))
    }

    // calculate the maximum number of vectors that can be stored in the given memory limit
    pub fn max_vectors(memory_limit: usize, dim: usize) -> usize {
        memory_limit / (dim * std::mem::size_of::<f32>() + std::mem::size_of::<VectorId>())
    }

    /// Iterate (id, vector_slice) over all entries.
    pub fn iter(&self) -> impl Iterator<Item = (VectorId, &[f32])> {
        (0..self.len).map(move |i| self.get(i).unwrap())
    }

    /// The raw contiguous float buffer — useful for SIMD batch ops.
    pub fn as_flat_slice(&self) -> &[f32] {
        // SAFETY: allocation covers dim * capacity floats.
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.len * self.dim) }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl std::fmt::Debug for VectorBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorBlock")
            .field("dim", &self.dim)
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .finish()
    }
}

pub struct VectorStore {
    blocks: Vec<VectorBlock>,
    num_vectors_per_block: usize,
    arena_capacity: usize,
}

impl VectorStore {
    pub fn new(dim: usize, arena_capacity: usize) -> Self {
        let num_vectors_per_block = VectorBlock::max_vectors(arena_capacity, dim);
        Self {
            blocks: Vec::new(),
            num_vectors_per_block,
            arena_capacity,
        }
    }

    pub fn insert(&mut self, id: VectorId, vector_data: &[f32]) -> Option<InternalId> {
        if self.blocks.is_empty() {
            self.blocks.push(VectorBlock::try_new(
                vector_data.len(),
                self.arena_capacity,
            )?);
        }
        let mut is_new_block = false;
        loop {
            if let Some(last) = self.blocks.last_mut() {
                if let Some(idx) = last.push(id, vector_data) {
                    return Some(Vector::calculate_internal_id(
                        self.blocks.len() - 1,
                        idx,
                        self.num_vectors_per_block,
                    ));
                } else {
                    if is_new_block {
                        return None;
                    }
                    self.blocks.push(VectorBlock::try_new(
                        vector_data.len(),
                        self.arena_capacity,
                    )?);
                    is_new_block = true;
                }
            }
        }
    }

    #[must_use]
    pub fn get(&self, id: InternalId) -> Option<(VectorId, &[f32])> {
        let block_index = Vector::derive_block_index(id, self.num_vectors_per_block);
        let vector_index = Vector::derive_vector_index(id, self.num_vectors_per_block);
        return self
            .blocks
            .get(block_index)
            .and_then(|block| block.get(vector_index));
    }

    /// All `(id, vector slice)` pairs in the store. Order is not specified.
    pub fn iter(&self) -> impl Iterator<Item = (VectorId, &[f32])> + '_ {
        self.blocks.iter().flat_map(|block| block.iter())
    }

    #[must_use]
    pub fn num_vectors(&self) -> usize {
        self.blocks.iter().map(|block| block.len()).sum()
    }

    #[must_use]
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Import up to `limit` vectors from `.fvecs` layout (`dim` must match each record).
    pub fn import_fvecs(&mut self, data: &[u8], dim: usize, limit: usize) -> usize {
        common::import_fvecs(data, dim, limit, |i, buf| {
            self.insert(VectorId(i), buf).is_some()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{fvecs_vector_count, import_fvecs, read_fvecs_dim_le, DEFAULT_ARENA_CAPACITY};
    use memmap2::Mmap;
    use std::fs::File;

    const SIFT_DIM: usize = 128;

    #[test]
    fn vector_block_push_and_get() {
        let mut block = VectorBlock::try_new(4, DEFAULT_ARENA_CAPACITY).expect("test alloc");
        block.push(VectorId(0), &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let (id, data) = block.get(0).unwrap();
        assert_eq!(id, VectorId(0));
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn store_arena_allocation_is_correct() {
        // Match [`VectorBlock::try_new`]'s mmap cap (`DEFAULT_ARENA_CAPACITY`).
        let memory_limit = DEFAULT_ARENA_CAPACITY;
        for dim in [4, 8, 16, 32, 64, 128] {
            let mut block = VectorBlock::try_new(dim, memory_limit).expect("test alloc");
            let max_vectors = VectorBlock::max_vectors(memory_limit, dim);
            for i in 0..max_vectors {
                let row = vec![1.0 * i as f32; dim];
                assert!(block.push(VectorId(i as u64), &row).is_some());
            }
            assert_eq!(block.len(), max_vectors);
            assert_eq!(block.capacity(), max_vectors);
            assert_eq!(block.as_flat_slice().len(), max_vectors * dim);

            let row = vec![1.0 * max_vectors as f32; dim];
            assert!(block.push(VectorId(max_vectors as u64), &row).is_none());
        }
    }

    #[test]
    fn test_data_not_overlapping() {
        for i in 0..100 {
            let mut block = VectorBlock::try_new(128, DEFAULT_ARENA_CAPACITY).expect("test alloc");
            block
                .push(VectorId(i as u64), &[1.0 * i as f32; 128])
                .unwrap();
            let (id, data) = block.get(0).unwrap();
            assert_eq!(id, VectorId(i as u64));
            assert_eq!(data, &[1.0 * i as f32; 128]);
        }
    }

    #[test]
    fn vector_block_full_returns_none() {
        let dim = 4;
        let mut block = VectorBlock::try_new(4, dim * size_of::<f32>() + size_of::<VectorId>())
            .expect("test alloc");
        block.push(VectorId(0), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(block.push(VectorId(1), &[0.0, 1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_vector_store_insert() {
        let dim = 128;
        let mut vector_store = VectorStore::new(dim, DEFAULT_ARENA_CAPACITY);

        let num_blocks = 3;
        while vector_store.num_blocks() < num_blocks {
            let id = VectorId(vector_store.num_vectors() as u64);
            let vector_data = vec![1.0 * id.0 as f32; dim];
            let internal_id = vector_store.insert(id, &vector_data).unwrap();
            let (vector_id, data) = vector_store.get(internal_id).unwrap();
            assert_eq!(vector_id, id);
            assert_eq!(data, &vector_data);
        }
    }

    #[test]
    fn test_sift1m_import() {
        let base_path = match std::env::var("SIFT1M_BASE_PATH") {
            Ok(path) => path,
            Err(_) => {
                eprintln!("SIFT1M_BASE_PATH not set");
                return;
            }
        };

        let path = std::path::PathBuf::from(base_path + "/sift_base.fvecs");

        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("could not open file {path:?}: {e}");
                return;
            }
        };
        let mmap = match unsafe { Mmap::map(&file) } {
            Ok(m) => m,
            Err(e) => {
                eprintln!("mmap failed for {path:?}: {e}");
                return;
            }
        };

        let limit: usize = std::env::var("SIFT1M_LIMIT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000_000);
        let data: &[u8] = &mmap[..];
        let Some(dim0) = read_fvecs_dim_le(data, 0) else {
            eprintln!("empty or truncated fvecs at {path:?}");
            return;
        };
        if dim0 != SIFT_DIM {
            eprintln!(
                "warning: first vector dim is {dim0}, expected {SIFT_DIM} for standard SIFT; using {dim0}"
            );
        }
        let avail = fvecs_vector_count(data, dim0);
        if avail == 0 {
            eprintln!("no complete vectors in {path:?}");
            return;
        }
        let use_limit = limit.min(avail);
        eprintln!(
            "SIFT1M_BASE={}  vectors_available={avail}  using_limit={use_limit}  dim={dim0}",
            path.display()
        );
        let mut store = VectorStore::new(dim0, DEFAULT_ARENA_CAPACITY);
        let n = import_fvecs(data, dim0, use_limit, |i, buf| {
            store.insert(VectorId(i), buf).is_some()
        });
        assert_eq!(
            n, use_limit,
            "full uniform .fvecs should import up to the chosen limit"
        );
    }
}
