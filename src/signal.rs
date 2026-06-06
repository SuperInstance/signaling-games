//! Signal: messages from sender to receiver.
//!
//! A signal is an observable action taken by the sender to convey (or conceal)
//! their private type. Signal cost may depend on the sender's type — this is
//! the core mechanism that enables separating equilibria.

use serde::{Deserialize, Serialize};

/// A named signal that a sender can emit.
///
/// # Examples
///
/// ```
/// use signaling_games::signal::Signal;
///
/// let s = Signal::new("education".into(), 3.0);
/// assert_eq!(s.label(), "education");
/// assert!((s.base_cost() - 3.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Signal {
    label: String,
    base_cost: f64,
}

impl Signal {
    /// Create a new signal with a label and base cost.
    pub fn new(label: String, base_cost: f64) -> Self {
        Self { label, base_cost }
    }

    /// The human-readable name of this signal.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Base cost of sending this signal (before type-dependent adjustment).
    pub fn base_cost(&self) -> f64 {
        self.base_cost
    }

    /// Compute the cost of this signal for a sender of a given type.
    ///
    /// In Spence-style models, high-ability senders pay lower marginal cost
    /// for education. The `type_cost_factor` modulates the base cost:
    /// `cost = base_cost * type_cost_factor`.
    ///
    /// A lower `type_cost_factor` means the type finds signaling cheaper.
    pub fn cost_for_type(&self, type_cost_factor: f64) -> f64 {
        self.base_cost * type_cost_factor
    }
}

/// The set of available signals a sender can choose from.
///
/// # Examples
///
/// ```
/// use signaling_games::signal::{Signal, SignalSpace};
///
/// let space = SignalSpace::new(vec![
///     Signal::new("no_degree".into(), 0.0),
///     Signal::new("bachelors".into(), 2.0),
///     Signal::new("phd".into(), 5.0),
/// ]);
/// assert_eq!(space.len(), 3);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SignalSpace {
    signals: Vec<Signal>,
}

impl SignalSpace {
    /// Create a signal space from a list of signals.
    pub fn new(signals: Vec<Signal>) -> Self {
        Self { signals }
    }

    /// Number of available signals.
    pub fn len(&self) -> usize {
        self.signals.len()
    }

    /// True if there are no signals.
    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }

    /// Iterate over available signals.
    pub fn iter(&self) -> impl Iterator<Item = &Signal> {
        self.signals.iter()
    }

    /// Get a signal by index.
    pub fn get(&self, index: usize) -> Option<&Signal> {
        self.signals.get(index)
    }

    /// All signals as a slice.
    pub fn as_slice(&self) -> &[Signal] {
        &self.signals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_new() {
        let s = Signal::new("hello".into(), 1.5);
        assert_eq!(s.label(), "hello");
        assert!((s.base_cost() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn signal_cost_for_type() {
        let s = Signal::new("edu".into(), 4.0);
        // High-ability type: cost factor 0.5 → cost = 2.0
        assert!((s.cost_for_type(0.5) - 2.0).abs() < 1e-10);
        // Low-ability type: cost factor 1.0 → cost = 4.0
        assert!((s.cost_for_type(1.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn signal_space_basics() {
        let space = SignalSpace::new(vec![
            Signal::new("a".into(), 1.0),
            Signal::new("b".into(), 2.0),
        ]);
        assert_eq!(space.len(), 2);
        assert!(!space.is_empty());
        assert_eq!(space.get(0).unwrap().label(), "a");
        assert_eq!(space.get(1).unwrap().label(), "b");
        assert!(space.get(2).is_none());
    }

    #[test]
    fn signal_space_empty() {
        let space = SignalSpace::new(vec![]);
        assert!(space.is_empty());
        assert_eq!(space.len(), 0);
    }
}
