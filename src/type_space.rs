//! Type space: the sender's private type and prior distribution.
//!
//! The sender has a private type (hidden information) drawn from a prior
//! distribution. The receiver does not observe the type directly — they
//! must infer it from the signal.

use serde::{Deserialize, Serialize};

/// A sender type with a name and cost factor for signaling.
///
/// The cost factor determines how expensive signaling is for this type.
/// In Spence's job market model, high-ability workers have a lower cost
/// factor for education.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SenderType {
    name: String,
    cost_factor: f64,
}

impl SenderType {
    /// Create a new sender type.
    ///
    /// - `name`: human-readable label (e.g., "high", "low")
    /// - `cost_factor`: multiplier on signal base cost (lower = cheaper signaling)
    pub fn new(name: String, cost_factor: f64) -> Self {
        Self { name, cost_factor }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn cost_factor(&self) -> f64 {
        self.cost_factor
    }
}

/// A type space: the set of possible sender types and their prior probabilities.
///
/// Priors must be non-negative and are normalized internally so they sum to 1.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TypeSpace {
    types: Vec<SenderType>,
    priors: Vec<f64>,
}

impl TypeSpace {
    /// Create a new type space.
    ///
    /// Priors are normalized if they don't sum to 1.
    pub fn new(types: Vec<SenderType>, priors: Vec<f64>) -> Self {
        assert_eq!(
            types.len(),
            priors.len(),
            "types and priors must have same length"
        );
        assert!(!types.is_empty(), "type space must have at least one type");
        let sum: f64 = priors.iter().sum();
        assert!(sum > 0.0, "priors must sum to a positive number");
        let priors: Vec<f64> = priors.iter().map(|p| p / sum).collect();
        Self { types, priors }
    }

    /// Number of types.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Get a type by index.
    pub fn get_type(&self, index: usize) -> Option<&SenderType> {
        self.types.get(index)
    }

    /// Get the prior probability of the type at `index`.
    pub fn prior(&self, index: usize) -> f64 {
        self.priors[index]
    }

    /// Iterate over (type, prior) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&SenderType, f64)> {
        self.types.iter().zip(self.priors.iter().copied())
    }

    /// All types as a slice.
    pub fn types(&self) -> &[SenderType] {
        &self.types
    }

    /// All priors as a slice.
    pub fn priors(&self) -> &[f64] {
        &self.priors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sender_type_basic() {
        let t = SenderType::new("high".into(), 0.5);
        assert_eq!(t.name(), "high");
        assert!((t.cost_factor() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn type_space_two_types() {
        let ts = TypeSpace::new(
            vec![
                SenderType::new("high".into(), 0.5),
                SenderType::new("low".into(), 1.0),
            ],
            vec![0.5, 0.5],
        );
        assert_eq!(ts.len(), 2);
        assert!((ts.prior(0) - 0.5).abs() < 1e-10);
        assert!((ts.prior(1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn type_space_normalizes_priors() {
        let ts = TypeSpace::new(
            vec![
                SenderType::new("a".into(), 1.0),
                SenderType::new("b".into(), 1.0),
            ],
            vec![3.0, 7.0],
        );
        assert!((ts.prior(0) - 0.3).abs() < 1e-10);
        assert!((ts.prior(1) - 0.7).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn type_space_mismatched_lengths() {
        TypeSpace::new(vec![SenderType::new("a".into(), 1.0)], vec![0.5, 0.5]);
    }

    #[test]
    #[should_panic(expected = "positive number")]
    fn type_space_zero_priors() {
        TypeSpace::new(vec![SenderType::new("a".into(), 1.0)], vec![0.0]);
    }
}
