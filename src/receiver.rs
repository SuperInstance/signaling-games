//! Receiver: observes signals, forms posterior beliefs, chooses actions.
//!
//! The receiver watches the sender's signal, updates beliefs using Bayes' rule,
//! and selects the action that maximizes expected utility given the posterior.

use serde::{Deserialize, Serialize};

use crate::bayes::BayesianUpdate;
use crate::signal::Signal;
use crate::type_space::TypeSpace;

/// A named action the receiver can take.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Action {
    label: String,
}

impl Action {
    pub fn new(label: String) -> Self {
        Self { label }
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Receiver payoff: (type_index, action_index) → payoff to receiver.
///
/// Encoded as a flat map from `(usize, usize)` to `f64`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReceiverPayoff {
    /// n_types × n_actions payoff matrix, row-major.
    matrix: Vec<Vec<f64>>,
}

impl ReceiverPayoff {
    /// Create a payoff matrix.
    ///
    /// `matrix[type_index][action_index]` = receiver's payoff when the sender
    /// is that type and the receiver chooses that action.
    pub fn new(matrix: Vec<Vec<f64>>) -> Self {
        Self { matrix }
    }

    /// Get the payoff for a given type and action.
    pub fn get(&self, type_idx: usize, action_idx: usize) -> f64 {
        self.matrix[type_idx][action_idx]
    }

    /// Number of types (rows).
    pub fn n_types(&self) -> usize {
        self.matrix.len()
    }

    /// Number of actions (columns), or 0 if empty.
    pub fn n_actions(&self) -> usize {
        self.matrix.first().map_or(0, |r| r.len())
    }
}

/// The receiver in a signaling game.
///
/// Observes a signal, updates beliefs, and chooses the best action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Receiver {
    /// Available actions.
    actions: Vec<Action>,
    /// Payoff structure.
    payoff: ReceiverPayoff,
}

impl Receiver {
    /// Create a new receiver with actions and payoff matrix.
    pub fn new(actions: Vec<Action>, payoff: ReceiverPayoff) -> Self {
        assert_eq!(actions.len(), payoff.n_actions());
        Self { actions, payoff }
    }

    /// Available actions.
    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    /// Choose the best action given a posterior belief over types.
    ///
    /// Returns `(action_index, expected_payoff)`.
    pub fn best_response(&self, posterior: &[f64]) -> (usize, f64) {
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;

        for a in 0..self.payoff.n_actions() {
            let eu: f64 = posterior
                .iter()
                .enumerate()
                .map(|(t, &p)| p * self.payoff.get(t, a))
                .sum();
            if eu > best_val {
                best_val = eu;
                best_idx = a;
            }
        }

        (best_idx, best_val)
    }

    /// Observe a signal and compute the best response.
    ///
    /// Performs Bayesian update from prior, then chooses the best action.
    ///
    /// Returns `(posterior, best_action_index, expected_payoff)`.
    pub fn observe_signal(
        &self,
        type_space: &TypeSpace,
        _signal: &Signal,
        signal_costs: &[f64], // cost_factor per type for this signal
    ) -> (Vec<f64>, usize, f64) {
        let prior: Vec<f64> = type_space.priors().to_vec();

        // Likelihood: P(signal | type_i) ∝ exp(-cost_i)
        // Lower cost means more likely to send this signal
        let likelihoods: Vec<f64> = signal_costs.iter().map(|c| (-c).exp()).collect();

        let posterior = BayesianUpdate::update(&prior, &likelihoods);
        let (action_idx, payoff) = self.best_response(&posterior);

        (posterior, action_idx, payoff)
    }

    /// The payoff matrix.
    pub fn payoff(&self) -> &ReceiverPayoff {
        &self.payoff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_label() {
        let a = Action::new("hire".into());
        assert_eq!(a.label(), "hire");
    }

    #[test]
    fn payoff_matrix() {
        let p = ReceiverPayoff::new(vec![vec![10.0, 0.0], vec![0.0, 5.0]]);
        assert_eq!(p.n_types(), 2);
        assert_eq!(p.n_actions(), 2);
        assert!((p.get(0, 0) - 10.0).abs() < 1e-10);
        assert!((p.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn best_response_clear_preference() {
        let r = Receiver::new(
            vec![Action::new("hire".into()), Action::new("reject".into())],
            ReceiverPayoff::new(vec![
                vec![10.0, 0.0], // type 0: prefer hire
                vec![0.0, 10.0], // type 1: prefer reject
            ]),
        );
        // If certain sender is type 0 → hire
        let (idx, _) = r.best_response(&[1.0, 0.0]);
        assert_eq!(idx, 0);
        // If certain sender is type 1 → reject
        let (idx, _) = r.best_response(&[0.0, 1.0]);
        assert_eq!(idx, 1);
    }

    #[test]
    fn best_response_mixed_belief() {
        let r = Receiver::new(
            vec![Action::new("hire".into()), Action::new("reject".into())],
            ReceiverPayoff::new(vec![vec![10.0, 0.0], vec![0.0, 5.0]]),
        );
        // 50/50: E[hire] = 5, E[reject] = 2.5 → hire
        let (idx, eu) = r.best_response(&[0.5, 0.5]);
        assert_eq!(idx, 0);
        assert!((eu - 5.0).abs() < 1e-10);
    }
}
