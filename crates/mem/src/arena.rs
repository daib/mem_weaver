//! Bump arena over one anonymous memory map (`mmap`). On Linux, the mapping is advised with
//! `MADV_HUGEPAGE` so the kernel may back it with transparent huge pages when policy allows.
//!
//! [`Arena::try_with_capacity`] sets a **logical** byte cap (payload + alignment padding). The
//! mapping is rounded up to a multiple of the OS page size, so [`Arena::mapped_bytes`] may be
//! larger than [`Arena::capacity_bytes`].

use memmap2::MmapMut;
use std::alloc::Layout;
use std::cell::{Cell, UnsafeCell};
use std::io;
use std::mem::size_of;
use std::ptr::{self, NonNull};

pub struct Arena {
    max_bytes: usize,
    bump: Cell<usize>,
    /// `None` when `capacity_bytes == 0`.
    storage: UnsafeCell<Option<MmapMut>>,
}

impl Arena {
    /// Map anonymous memory (rounded up to a full page) and apply huge-page advice on Linux.
    /// Returns an error if the `mmap` fails (e.g. out of memory).
    pub fn try_with_capacity(capacity: usize) -> io::Result<Self> {
        if capacity == 0 {
            return Ok(Self {
                max_bytes: 0,
                bump: Cell::new(0),
                storage: UnsafeCell::new(None),
            });
        }

        let mapped_len = capacity.next_multiple_of(system_page_size());
        // Linux `madvise` uses `as_mut_ptr()` (`&mut self`); other targets skip that block, so avoid
        // `mut` there (silences `unused_mut`).
        #[cfg(target_os = "linux")]
        let mut map = MmapMut::map_anon(mapped_len)?;
        #[cfg(not(target_os = "linux"))]
        let map = MmapMut::map_anon(mapped_len)?;

        // Transparent huge pages (Linux): `MADV_HUGEPAGE` hint; ignore errors.
        #[cfg(target_os = "linux")]
        unsafe {
            extern "C" {
                fn madvise(addr: *mut core::ffi::c_void, len: usize, advice: i32) -> i32;
            }
            const MADV_HUGEPAGE: i32 = 14;
            let p = map.as_mut_ptr();
            let _ = madvise(p.cast(), map.len(), MADV_HUGEPAGE);
        }

        Ok(Self {
            max_bytes: capacity,
            bump: Cell::new(0),
            storage: UnsafeCell::new(Some(map)),
        })
    }

    #[must_use]
    pub fn capacity_bytes(&self) -> usize {
        self.max_bytes
    }

    pub fn allocated_bytes(&self) -> usize {
        self.bump.get()
    }

    #[must_use]
    pub fn remaining_bytes(&self) -> usize {
        self.max_bytes.saturating_sub(self.bump.get())
    }

    /// Bytes reserved from the kernel for this arena (multiple of page size, or 0).
    #[must_use]
    pub fn mapped_bytes(&self) -> usize {
        unsafe { (*self.storage.get()).as_ref().map_or(0, |m| m.len()) }
    }

    /// Alias for [`Arena::mapped_bytes`] (legacy name from the bumpalo era).
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.mapped_bytes()
    }

    /// Base pointer of the mmap backing store (bump offset `0`). Null when [`Arena::capacity_bytes`] is `0`.
    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        unsafe {
            (*self.storage.get())
                .as_ref()
                .map_or(ptr::null(), |m| m.as_ptr())
        }
    }

    pub fn try_alloc_vector(&self, dim: usize) -> Option<&mut [f32]> {
        let layout = Layout::from_size_align(dim * size_of::<f32>(), 32).ok()?;
        let start = self.alloc_raw(layout)?;
        let ptr = start.as_ptr().cast::<f32>();
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, dim) })
    }

    pub fn try_alloc_vector_aligned(&self, dim: usize, align: usize) -> Option<&mut [f32]> {
        let layout = Layout::from_size_align(dim * size_of::<f32>(), align).ok()?;
        let start = self.alloc_raw(layout)?;
        let ptr = start.as_ptr().cast::<f32>();
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, dim) })
    }

    pub fn try_alloc<T: Copy>(&self, val: T) -> Option<&mut T> {
        if size_of::<T>() == 0 {
            let ptr = NonNull::dangling().as_ptr();
            return Some(unsafe { &mut *ptr });
        }
        let layout = Layout::new::<T>();
        let start = self.alloc_raw(layout)?;
        unsafe {
            let p = start.cast::<T>().as_ptr();
            ptr::write(p, val);
            Some(&mut *p)
        }
    }

    /// Contiguous `[T; len]` in the arena, all bits zero. `T` must be valid when all-zero
    /// (e.g. `VectorId(0)`).
    pub fn try_alloc_slice<T: Copy>(&self, len: usize) -> Option<&mut [T]> {
        if len == 0 {
            return Some(&mut []);
        }
        let layout = Layout::array::<T>(len).ok()?;
        let ptr_u8 = self.alloc_raw(layout)?;
        let p = ptr_u8.as_ptr().cast::<T>();
        unsafe { Some(std::slice::from_raw_parts_mut(p, len)) }
    }

    /// Contiguous `[T; len]` in the arena, all bits zero. `T` must be valid when all-zero
    /// (e.g. `VectorId(0)`).
    pub fn try_alloc_slice_zeroed<T: Copy>(&self, len: usize) -> Option<&mut [T]> {
        if len == 0 {
            return Some(&mut []);
        }
        let layout = Layout::array::<T>(len).ok()?;
        let ptr_u8 = self.alloc_raw(layout)?;
        let p = ptr_u8.as_ptr().cast::<T>();
        unsafe {
            ptr::write_bytes(p, 0, len);
            Some(std::slice::from_raw_parts_mut(p, len))
        }
    }

    /// Contiguous `[T; len]` in the arena, all bits zero. `T` must be valid when all-zero
    /// (e.g. `VectorId(0)`).
    pub fn try_alloc_slice_aligned<T: Copy>(&self, len: usize, align: usize) -> Option<&mut [T]> {
        if len == 0 {
            return Some(&mut []);
        }
        let layout = Layout::from_size_align(len * size_of::<T>(), align).ok()?;
        let ptr_u8 = self.alloc_raw(layout)?;
        let p = ptr_u8.as_ptr().cast::<T>();
        unsafe { Some(std::slice::from_raw_parts_mut(p, len)) }
    }

    /// Copy `src` into a contiguous arena slice.
    pub fn try_alloc_slice_copy<T: Copy>(&self, src: &[T]) -> Option<&mut [T]> {
        if src.is_empty() {
            return Some(&mut []);
        }
        let layout = Layout::array::<T>(src.len()).ok()?;
        let ptr_u8 = self.alloc_raw(layout)?;
        let p = ptr_u8.as_ptr().cast::<T>();
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), p, src.len());
            Some(std::slice::from_raw_parts_mut(p, src.len()))
        }
    }

    /// Uninitialized `[T; count]` in the arena. Caller must initialize (e.g. `copy_nonoverlapping`)
    /// before reading as `T`.
    pub fn try_alloc_array_ptr<T>(&self, count: usize) -> Option<NonNull<T>> {
        if count == 0 {
            return Some(NonNull::dangling());
        }
        let layout = Layout::array::<T>(count).ok()?;
        let ptr_u8 = self.alloc_raw(layout)?;
        Some(ptr_u8.cast())
    }

    pub fn reset(&mut self) {
        self.bump.set(0);
    }

    fn alloc_raw(&self, layout: Layout) -> Option<NonNull<u8>> {
        let align = layout.align().max(1);
        let size = layout.size();
        let current = self.bump.get();
        let start = align_up(current, align)?;
        let new_bump = start.checked_add(size)?;
        if new_bump > self.max_bytes {
            return None;
        }

        unsafe {
            let storage = &mut *self.storage.get();
            let map = storage.as_mut()?;
            let base = map.as_mut_ptr();
            if start > current {
                let pad = start - current;
                ptr::write_bytes(base.add(current), 0, pad);
            }
            let ptr = base.add(start);
            self.bump.set(new_bump);
            Some(NonNull::new_unchecked(ptr))
        }
    }
}

impl std::fmt::Debug for Arena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Arena")
            .field("capacity_bytes", &self.max_bytes)
            .field("allocated_bytes", &self.allocated_bytes())
            .field("mapped_bytes", &self.mapped_bytes())
            .finish()
    }
}

fn align_up(n: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    Some(n.checked_add(align - 1)? & !(align - 1))
}

/// Conservative default used to round mapping length to a whole number of pages.
/// (Avoids a direct `libc` dependency; `mem` stays usable when the host page size differs.)
fn system_page_size() -> usize {
    #[cfg(target_vendor = "apple")]
    {
        16 * 1024
    }
    #[cfg(all(unix, not(target_vendor = "apple")))]
    {
        4096
    }
    #[cfg(not(unix))]
    {
        4096
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_vector_is_aligned() {
        let arena = Arena::try_with_capacity(2 * 1024 * 1024).expect("test mmap");
        let v = arena.try_alloc_vector(128).expect("test alloc");
        assert_eq!(v.len(), 128);
        assert_eq!(v.as_ptr() as usize % 32, 0, "must be 32-byte aligned");
    }

    #[test]
    fn multiple_allocs_do_not_overlap() {
        let arena = Arena::try_with_capacity(4 * 1024 * 1024).expect("test mmap");
        let a = arena.try_alloc_vector(128).expect("test alloc");
        let b = arena.try_alloc_vector(128).expect("test alloc");
        assert_ne!(a.as_ptr(), b.as_ptr());
    }

    #[test]
    fn reset_restarts_bump_in_chunk() {
        let mut arena = Arena::try_with_capacity(1024 * 1024).expect("test mmap");
        let p1 = arena.try_alloc_vector(128).expect("test alloc").as_ptr();
        arena.reset();
        let p2 = arena.try_alloc_vector(128).expect("test alloc").as_ptr();
        assert_eq!(
            p1, p2,
            "after reset, allocations should start from the same mapping base"
        );
    }

    #[test]
    fn alloc_stores_value_and_aligns() {
        let arena = Arena::try_with_capacity(1024).expect("test mmap");
        let p = arena.try_alloc(0xfeed_face_u64).expect("test alloc");
        assert_eq!(*p, 0xfeed_face);
        assert_eq!(
            p as *mut u64 as usize % std::mem::align_of::<u64>(),
            0,
            "alloc should respect alignment"
        );
        *p = 1;
        assert_eq!(*p, 1);
    }

    #[test]
    fn alloc_multiple_distinct_slots() {
        let arena = Arena::try_with_capacity(1024).expect("test mmap");
        let a = arena.try_alloc(10u32).expect("test alloc");
        let b = arena.try_alloc(20u32).expect("test alloc");
        assert_eq!(*a, 10);
        assert_eq!(*b, 20);
        assert_ne!(a as *mut u32, b as *mut u32);
    }

    #[test]
    fn alloc_vector_returns_none_when_exceeding_capacity() {
        let arena = Arena::try_with_capacity(32).expect("test mmap");
        assert!(
            arena.try_alloc_vector(4096).is_none(),
            "logical usage must not exceed capacity_bytes"
        );
        assert_eq!(arena.allocated_bytes(), 0);
    }

    #[test]
    fn second_alloc_returns_none_when_no_room() {
        let arena = Arena::try_with_capacity(256).expect("test mmap");
        assert!(arena.try_alloc_vector(16).is_some());
        assert!(
            arena.try_alloc_vector(64).is_none(),
            "cumulative logical usage must stay within capacity_bytes"
        );
    }

    #[test]
    fn mapped_size_rounds_up_to_page() {
        let arena = Arena::try_with_capacity(1).expect("test mmap");
        assert!(arena.mapped_bytes() >= system_page_size());
        assert_eq!(arena.capacity_bytes(), 1);
    }

    #[test]
    fn zero_capacity_has_no_mapping() {
        let arena = Arena::try_with_capacity(0).expect("test mmap");
        assert_eq!(arena.mapped_bytes(), 0);
        assert!(arena.try_alloc_vector(1).is_none());
    }
}
