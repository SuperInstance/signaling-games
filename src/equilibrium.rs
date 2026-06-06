//! Equilibrium: Perfect Bayesian Equilibrium finder for signaling games.
//!
//! A Perfect Bayesian Equilibrium (PBE) consists of:
//! 1. A **strategy** for the sender: mapping from type → signal
//! 2. A **strategy** for the receiver: mapping from signal → action
//! 3. A **belief system**: posterior over types after each signal
//!
//! such that:
//! - The sender's strategy is optimal given the receiver's response
//! - The receiver's strategy is optimal given their beliefs
//! - Beliefs are consistent with the strategies via Bayes' rule

use serde::{Deserialize, Serialize};

use crate::receiver::Receiver;
use crate::signal::SignalSpace;
use crate::type_space::TypeSpace;

/// Classification of equilibrium type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EquilibriumKind {
    /// All types send the same signal.
    Pooling,
    /// Each type sends a distinct signal.
    Separating,
    /// Some types mix between signals.
    SemiSeparating,
}

/// A sender strategy: type_index → signal_index.
pub type SenderStrategy = Vec<usize>;

/// A receiver strategy: signal_index → action_index.
pub type ReceiverStrategy = Vec<usize>;

/// Beliefs after observing each signal: signal_index → posterior over types.
pub type BeliefSystem = Vec<Vec<f64>>;

/// A Perfect Bayesian Equilibrium.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Equilibrium {
    /// Classification of this equilibrium.
    pub kind: EquilibriumKind,
    /// Sender's strategy: type_index → signal_index chosen.
    pub sender_strategy: SenderStrategy,
    /// Receiver's strategy: signal_index → action_index chosen.
    pub receiver_strategy: ReceiverStrategy,
    /// Posterior beliefs: signal_index → Vec<f64> (prob per type).
    pub beliefs: BeliefSystem,
    /// Expected payoff for each sender type under this equilibrium.
    pub sender_payoffs: Vec<f64>,
    /// Expected payoff for the receiver under this equilibrium.
    pub receiver_payoff: f64,
}

/// Computes sender payoff given their type, chosen signal, and receiver response.
///
/// `sender_type_payoff[type_idx][action_idx]` = sender's utility when receiver
/// takes that action, minus the signaling cost.
pub fn sender_utility(
    type_idx: usize,
    _signal_idx: usize,
    action_idx: usize,
    signal_cost: f64,
    sender_type_payoff: &[Vec<f64>],
) -> f64 {
    sender_type_payoff[type_idx][action_idx] - signal_cost
}

/// Finder for Perfect Bayesian Equilibria.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EquilibriumFinder {
    type_space: TypeSpace,
    signal_space: SignalSpace,
    receiver: Receiver,
    /// Sender payoff: type_idx → action_idx → payoff (excluding signal cost).
    sender_payoff_matrix: Vec<Vec<f64>>,
}

impl EquilibriumFinder {
    /// Create a new equilibrium finder.
    pub fn new(
        type_space: TypeSpace,
        signal_space: SignalSpace,
        receiver: Receiver,
        sender_payoff_matrix: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            type_space,
            signal_space,
            receiver,
            sender_payoff_matrix,
        }
    }

    /// Classify a sender strategy.
    pub fn classify_strategy(&self, strategy: &SenderStrategy) -> EquilibriumKind {
        let unique: std::collections::HashSet<usize> = strategy.iter().copied().collect();
        if unique.len() == 1 {
            EquilibriumKind::Pooling
        } else if unique.len() == strategy.len() {
            EquilibriumKind::Separating
        } else {
            EquilibriumKind::SemiSeparating
        }
    }

    /// Compute beliefs for a given sender strategy via Bayes' rule.
    ///
    /// For each signal, compute the posterior over types given which types
    /// send that signal and the prior distribution.
    pub fn compute_beliefs(&self, strategy: &SenderStrategy) -> BeliefSystem {
        let n_signals = self.signal_space.len();
        let n_types = self.type_space.len();

        let mut beliefs = vec![vec![0.0; n_types]; n_signals];

        // For each signal, which types send it?
        for (type_idx, &sig_idx) in strategy.iter().enumerate() {
            beliefs[sig_idx][type_idx] = self.type_space.prior(type_idx);
        }

        // Normalize each signal's belief vector
        for belief in beliefs.iter_mut() {
            let sum: f64 = belief.iter().sum();
            if sum > 0.0 {
                for b in belief.iter_mut() {
                    *b /= sum;
                }
            } else {
                // Off-path: uniform beliefs
                let uniform = 1.0 / n_types as f64;
                for b in belief.iter_mut() {
                    *b = uniform;
                }
            }
        }

        beliefs
    }

    /// Check if a sender strategy is a best response to the receiver strategy.
    ///
    /// For each type, verify that the assigned signal maximizes the sender's
    /// expected utility given the receiver's response.
    pub fn is_sender_best_response(
        &self,
        strategy: &SenderStrategy,
        receiver_strategy: &ReceiverStrategy,
        _beliefs: &BeliefSystem,
        epsilon: f64,
    ) -> bool {
        for (type_idx, &current_signal) in strategy.iter().enumerate() {
            let signal = self.signal_space.get(current_signal).unwrap();
            let cost =
                signal.cost_for_type(self.type_space.get_type(type_idx).unwrap().cost_factor());
            let action_idx = receiver_strategy[current_signal];
            let current_utility = sender_utility(
                type_idx,
                current_signal,
                action_idx,
                cost,
                &self.sender_payoff_matrix,
            );

            // Check all alternative signals
            for (alt_signal_idx, _) in receiver_strategy.iter().enumerate() {
                if alt_signal_idx == current_signal {
                    continue;
                }
                let alt_signal = self.signal_space.get(alt_signal_idx).unwrap();
                let alt_cost = alt_signal
                    .cost_for_type(self.type_space.get_type(type_idx).unwrap().cost_factor());

                // Receiver's response to alternative signal (off-path)
                let alt_action = receiver_strategy[alt_signal_idx];
                let alt_utility = sender_utility(
                    type_idx,
                    alt_signal_idx,
                    alt_action,
                    alt_cost,
                    &self.sender_payoff_matrix,
                );

                if alt_utility > current_utility + epsilon {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the receiver's best response for each signal given beliefs.
    pub fn compute_receiver_strategy(&self, beliefs: &BeliefSystem) -> ReceiverStrategy {
        let mut strategy = Vec::with_capacity(self.signal_space.len());
        for belief in beliefs {
            let (action_idx, _) = self.receiver.best_response(belief);
            strategy.push(action_idx);
        }
        strategy
    }

    /// Find all pure-strategy equilibria by exhaustive enumeration.
    ///
    /// Tries every possible sender strategy and checks equilibrium conditions.
    /// Returns all equilibria found.
    pub fn find_equilibria(&self, epsilon: f64) -> Vec<Equilibrium> {
        let n_types = self.type_space.len();
        let n_signals = self.signal_space.len();
        let mut results = Vec::new();

        // Enumerate all possible sender strategies
        let total = n_signals.pow(n_types as u32);
        for code in 0..total {
            let strategy = self.decode_strategy(code, n_types, n_signals);
            let beliefs = self.compute_beliefs(&strategy);
            let receiver_strat = self.compute_receiver_strategy(&beliefs);

            if self.is_sender_best_response(&strategy, &receiver_strat, &beliefs, epsilon) {
                let kind = self.classify_strategy(&strategy);

                // Compute payoffs
                let sender_payoffs: Vec<f64> = (0..n_types)
                    .map(|t| {
                        let sig = strategy[t];
                        let sig_obj = self.signal_space.get(sig).unwrap();
                        let cost = sig_obj
                            .cost_for_type(self.type_space.get_type(t).unwrap().cost_factor());
                        let action = receiver_strat[sig];
                        sender_utility(t, sig, action, cost, &self.sender_payoff_matrix)
                    })
                    .collect();

                // Receiver expected payoff
                let mut receiver_payoff = 0.0;
                for (t, &sig) in strategy.iter().enumerate() {
                    let action = receiver_strat[sig];
                    receiver_payoff +=
                        self.type_space.prior(t) * self.receiver.payoff().get(t, action);
                }

                results.push(Equilibrium {
                    kind,
                    sender_strategy: strategy,
                    receiver_strategy: receiver_strat,
                    beliefs,
                    sender_payoffs,
                    receiver_payoff,
                });
            }
        }

        results
    }

    /// Decode an integer into a sender strategy.
    fn decode_strategy(&self, code: usize, n_types: usize, n_signals: usize) -> SenderStrategy {
        let mut strategy = Vec::with_capacity(n_types);
        let mut remaining = code;
        for _ in 0..n_types {
            strategy.push(remaining % n_signals);
            remaining /= n_signals;
        }
        strategy
    }

    /// Convenience: find only pooling equilibria.
    pub fn find_pooling(&self, epsilon: f64) -> Vec<Equilibrium> {
        self.find_equilibria(epsilon)
            .into_iter()
            .filter(|e| e.kind == EquilibriumKind::Pooling)
            .collect()
    }

    /// Convenience: find only separating equilibria.
    pub fn find_separating(&self, epsilon: f64) -> Vec<Equilibrium> {
        self.find_equilibria(epsilon)
            .into_iter()
            .filter(|e| e.kind == EquilibriumKind::Separating)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receiver::{Action, Receiver, ReceiverPayoff};
    use crate::signal::{Signal, SignalSpace};
    use crate::type_space::{SenderType, TypeSpace};

    fn spence_setup() -> EquilibriumFinder {
        let type_space = TypeSpace::new(
            vec![
                SenderType::new("high".into(), 0.5),
                SenderType::new("low".into(), 1.5),
            ],
            vec![0.5, 0.5],
        );

        let signal_space = SignalSpace::new(vec![
            Signal::new("no_edu".into(), 0.0),
            Signal::new("edu".into(), 4.0),
        ]);

        let receiver = Receiver::new(
            vec![Action::new("manager".into()), Action::new("worker".into())],
            ReceiverPayoff::new(vec![
                vec![10.0, 2.0], // high type → manager best
                vec![2.0, 6.0],  // low type → worker best
            ]),
        );

        // Sender payoff (no signal cost): both types prefer manager
        let sender_payoff = vec![
            vec![10.0, 2.0], // high type payoff from (manager, worker)
            vec![8.0, 4.0],  // low type payoff from (manager, worker)
        ];

        EquilibriumFinder::new(type_space, signal_space, receiver, sender_payoff)
    }

    #[test]
    fn classify_pooling() {
        let finder = spence_setup();
        let kind = finder.classify_strategy(&vec![0, 0]); // both send signal 0
        assert_eq!(kind, EquilibriumKind::Pooling);
    }

    #[test]
    fn classify_separating() {
        let finder = spence_setup();
        let kind = finder.classify_strategy(&vec![0, 1]);
        assert_eq!(kind, EquilibriumKind::Separating);
    }

    #[test]
    fn classify_semi_separating_three_types() {
        let finder = EquilibriumFinder::new(
            TypeSpace::new(
                vec![
                    SenderType::new("a".into(), 1.0),
                    SenderType::new("b".into(), 1.0),
                    SenderType::new("c".into(), 1.0),
                ],
                vec![1.0, 1.0, 1.0],
            ),
            SignalSpace::new(vec![
                Signal::new("x".into(), 1.0),
                Signal::new("y".into(), 1.0),
            ]),
            Receiver::new(
                vec![Action::new("act".into())],
                ReceiverPayoff::new(vec![vec![0.0], vec![0.0], vec![0.0]]),
            ),
            vec![vec![0.0], vec![0.0], vec![0.0]],
        );
        // Types a and b send signal 0, type c sends signal 1
        let kind = finder.classify_strategy(&vec![0, 0, 1]);
        assert_eq!(kind, EquilibriumKind::SemiSeparating);
    }

    #[test]
    fn spence_find_equilibria() {
        let finder = spence_setup();
        let eqs = finder.find_equilibria(1e-6);
        assert!(
            !eqs.is_empty(),
            "Spence model should have at least one equilibrium"
        );
    }

    #[test]
    fn spence_separating_exists() {
        let finder = spence_setup();
        let sep = finder.find_separating(1e-6);
        // In Spence model, a separating equilibrium should exist where
        // high type gets education, low type doesn't
        assert!(
            sep.iter()
                .any(|e| e.sender_strategy[0] == 1 && e.sender_strategy[1] == 0),
            "Should find separating equilibrium: high=edu, low=no_edu"
        );
    }

    #[test]
    fn beliefs_pooling() {
        let finder = spence_setup();
        let beliefs = finder.compute_beliefs(&vec![0, 0]);
        // Pooling on signal 0: posterior = prior
        assert!((beliefs[0][0] - 0.5).abs() < 1e-10);
        assert!((beliefs[0][1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn beliefs_separating() {
        let finder = spence_setup();
        let beliefs = finder.compute_beliefs(&vec![1, 0]);
        // High sends signal 1, low sends signal 0
        assert!((beliefs[1][0] - 1.0).abs() < 1e-10); // signal 1 → certain high
        assert!((beliefs[0][1] - 1.0).abs() < 1e-10); // signal 0 → certain low
    }

    #[test]
    fn sender_utility_calculation() {
        // type 0, signal 0, action 0, cost = 1.0, payoff = 5.0
        let u = sender_utility(0, 0, 0, 1.0, &[vec![5.0]]);
        assert!((u - 4.0).abs() < 1e-10);
    }
}
