pub type NodeId = u32;
pub type InternalId = u32;
pub type ElementType = f32;

/// Stable external identifier for a stored vector row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VectorId(pub u64);

/// Distance metric for vector comparison (used by [`crate::distance`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    #[default]
    Cosine,
    DotProduct,
    Euclidean,
}
