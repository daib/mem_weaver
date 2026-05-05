//! Memory utilities: mmap-backed bump arena with optional Linux transparent huge page hints.
//!
//! See [`Arena::mapped_bytes`] vs [`Arena::capacity_bytes`].

mod arena;

pub use arena::Arena;
