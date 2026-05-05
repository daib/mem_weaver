//! ANN / retrieval evaluation helpers (e.g. recall@k).
use crate::VectorId;
use std::collections::HashSet;
use thiserror::Error;

/// Input or computed recall value failed validation.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum RecallValidationError {
    /// Recall must be a finite [0, 1] value (not NaN or ±∞).
    #[error("recall must be finite (not NaN or infinite)")]
    NotFinite,
    /// Recall must lie in the inclusive interval [0, 1].
    #[error("recall must be in [0, 1], got {0}")]
    OutOfRange(f32),
    /// No relevant ids to define recall against.
    #[error("ground truth must be non-empty for recall@k")]
    EmptyGroundTruth,
}

/// Checks that `recall` is a valid reported score: finite and in \[0, 1\].
pub fn validate_recall_score(recall: f32) -> Result<(), RecallValidationError> {
    if recall.is_nan() || recall.is_infinite() {
        return Err(RecallValidationError::NotFinite);
    }
    if recall < 0.0 || recall > 1.0 {
        return Err(RecallValidationError::OutOfRange(recall));
    }
    Ok(())
}

/// **Recall@k** (set-based): `| retrieved ∩ ground_truth | / | ground_truth |`,
/// where `ground_truth` is de-duplicated. `retrieved` is typically the top-`k` ANN ids in rank order
/// (order does not change the set intersection; duplicate ids in `retrieved` only count once toward
/// the intersection when matching the ground-truth set).
///
/// The returned value is checked with [`validate_recall_score`].
pub fn recall_at_k(
    retrieved: &[VectorId],
    ground_truth: &[VectorId],
) -> Result<f32, RecallValidationError> {
    let gt: HashSet<VectorId> = ground_truth.iter().copied().collect();
    if gt.is_empty() {
        return Err(RecallValidationError::EmptyGroundTruth);
    }
    let retrieved: HashSet<VectorId> = retrieved.iter().copied().collect();
    let hits = gt.intersection(&retrieved).count();
    let recall = hits as f32 / gt.len() as f32;
    validate_recall_score(recall)?;
    Ok(recall)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_endpoints() {
        assert!(validate_recall_score(0.0).is_ok());
        assert!(validate_recall_score(1.0).is_ok());
        assert!(validate_recall_score(0.5).is_ok());
    }

    #[test]
    fn validate_rejects_non_finite() {
        assert_eq!(
            validate_recall_score(f32::NAN).unwrap_err(),
            RecallValidationError::NotFinite
        );
        assert_eq!(
            validate_recall_score(f32::INFINITY).unwrap_err(),
            RecallValidationError::NotFinite
        );
    }

    #[test]
    fn validate_rejects_out_of_range() {
        assert!(matches!(
            validate_recall_score(-0.01),
            Err(RecallValidationError::OutOfRange(_))
        ));
        assert!(matches!(
            validate_recall_score(1.01),
            Err(RecallValidationError::OutOfRange(_))
        ));
    }

    #[test]
    fn recall_perfect_and_zero() {
        let gt = [VectorId(1), VectorId(2), VectorId(3)];
        assert_eq!(
            recall_at_k(&[VectorId(1), VectorId(2), VectorId(3)], &gt).unwrap(),
            1.0
        );
        assert_eq!(recall_at_k(&[VectorId(9), VectorId(8)], &gt).unwrap(), 0.0);
    }

    #[test]
    fn recall_partial() {
        let gt = [VectorId(1), VectorId(2), VectorId(3), VectorId(4)];
        let ret = [VectorId(1), VectorId(9), VectorId(2), VectorId(8)];
        let r = recall_at_k(&ret, &gt).unwrap();
        assert!((r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn recall_empty_ground_truth_errors() {
        assert_eq!(
            recall_at_k(&[VectorId(1)], &[]).unwrap_err(),
            RecallValidationError::EmptyGroundTruth
        );
    }
}
