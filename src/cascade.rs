//! Information cascade: sequential decision model where agents observe
//! predecessors' actions and may override their own private signal.
//!
//! In an information cascade, agents act sequentially. Each agent:
//! 1. Observes the actions of all previous agents
//! 2. Receives a private signal (binary: good/bad)
//! 3. Chooses an action based on both public history and private signal
//!
//! A cascade begins when the public evidence overwhelms the private signal,
//! making it rational to ignore private information.

use serde::{Deserialize, Serialize};

use crate::bayes::BayesianUpdate;

/// A binary private signal for an agent.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PrivateSignal {
    /// Positive signal (invest, adopt, accept).
    Good,
    /// Negative signal (reject, avoid, decline).
    Bad,
}

/// An agent's decision in the cascade.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CascadeDecision {
    /// Agent index (0-based).
    pub agent_index: usize,
    /// Private signal received.
    pub private_signal: PrivateSignal,
    /// Action chosen.
    pub action: CascadeAction,
    /// Whether this agent is in a cascade (ignoring private signal).
    pub in_cascade: bool,
    /// Posterior belief P(state = Good | history + signal).
    pub posterior: f64,
}

/// Action an agent can take.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CascadeAction {
    Adopt,
    Reject,
}

/// Result of running an information cascade simulation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CascadeResult {
    /// Each agent's decision in order.
    pub decisions: Vec<CascadeDecision>,
    /// Index where the cascade starts (if any).
    pub cascade_start: Option<usize>,
    /// Final proportion of adopters.
    pub adopt_rate: f64,
    /// Number of agents in cascade.
    pub cascade_count: usize,
}

/// Information cascade simulator.
///
/// Uses Bayesian updating to model sequential decision-making.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InformationCascade {
    /// Prior probability that the true state is Good.
    prior_good: f64,
    /// Signal accuracy: P(signal = Good | state = Good).
    accuracy: f64,
    /// Decision threshold: adopt if P(good) >= threshold.
    threshold: f64,
}

impl InformationCascade {
    /// Create a new cascade model.
    ///
    /// - `prior_good`: prior P(state = Good), in (0, 1)
    /// - `accuracy`: signal accuracy P(signal = correct | state), in (0.5, 1)
    /// - `threshold`: adopt if posterior >= threshold, in (0, 1)
    pub fn new(prior_good: f64, accuracy: f64, threshold: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&prior_good),
            "prior_good must be in (0, 1)"
        );
        assert!(
            (0.5..1.0).contains(&accuracy),
            "accuracy must be in (0.5, 1)"
        );
        assert!(
            (0.0..1.0).contains(&threshold),
            "threshold must be in (0, 1)"
        );
        Self {
            prior_good,
            accuracy,
            threshold,
        }
    }

    /// Compute the posterior belief P(state = Good | history of actions + private signal).
    ///
    /// Each action in history is treated as evidence: "adopt" is a positive signal,
    /// "reject" is a negative signal, both with accuracy derived from the model.
    fn posterior_after_history_and_signal(
        &self,
        history: &[CascadeAction],
        private_signal: PrivateSignal,
    ) -> f64 {
        // Build likelihoods for each observation
        let mut observations = Vec::new();

        // Each past action is evidence
        for &action in history {
            let likelihood = match action {
                CascadeAction::Adopt => vec![self.accuracy, 1.0 - self.accuracy],
                CascadeAction::Reject => vec![1.0 - self.accuracy, self.accuracy],
            };
            observations.push(likelihood);
        }

        // Add private signal as final observation
        let signal_likelihood = match private_signal {
            PrivateSignal::Good => vec![self.accuracy, 1.0 - self.accuracy],
            PrivateSignal::Bad => vec![1.0 - self.accuracy, self.accuracy],
        };
        observations.push(signal_likelihood);

        BayesianUpdate::sequential_update(&[self.prior_good, 1.0 - self.prior_good], &observations)
            [0]
    }

    /// Determine if an agent is in a cascade.
    ///
    /// An agent is in a cascade if their action would be the same regardless
    /// of their private signal.
    fn is_in_cascade(&self, history: &[CascadeAction]) -> bool {
        let post_good = self.posterior_after_history_and_signal(history, PrivateSignal::Good);
        let post_bad = self.posterior_after_history_and_signal(history, PrivateSignal::Bad);

        let action_good = if post_good >= self.threshold {
            CascadeAction::Adopt
        } else {
            CascadeAction::Reject
        };
        let action_bad = if post_bad >= self.threshold {
            CascadeAction::Adopt
        } else {
            CascadeAction::Reject
        };

        action_good == action_bad
    }

    /// Run the cascade simulation with given private signals.
    ///
    /// Agents decide sequentially. Each sees all prior actions.
    pub fn simulate(&self, signals: &[PrivateSignal]) -> CascadeResult {
        let mut decisions = Vec::with_capacity(signals.len());
        let mut history: Vec<CascadeAction> = Vec::new();
        let mut cascade_start: Option<usize> = None;
        let mut cascade_count = 0usize;

        for (i, &signal) in signals.iter().enumerate() {
            let in_cascade = self.is_in_cascade(&history);
            let posterior = self.posterior_after_history_and_signal(&history, signal);

            let action = if posterior >= self.threshold {
                CascadeAction::Adopt
            } else {
                CascadeAction::Reject
            };

            if in_cascade && cascade_start.is_none() {
                cascade_start = Some(i);
            }
            if in_cascade {
                cascade_count += 1;
            }

            decisions.push(CascadeDecision {
                agent_index: i,
                private_signal: signal,
                action,
                in_cascade,
                posterior,
            });

            history.push(action);
        }

        let adopt_count = decisions
            .iter()
            .filter(|d| d.action == CascadeAction::Adopt)
            .count();
        let adopt_rate = adopt_count as f64 / decisions.len().max(1) as f64;

        CascadeResult {
            decisions,
            cascade_start,
            adopt_rate,
            cascade_count,
        }
    }

    pub fn prior_good(&self) -> f64 {
        self.prior_good
    }

    pub fn accuracy(&self) -> f64 {
        self.accuracy
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_uniform_agreement() {
        // High accuracy, all good signals → everyone should adopt
        let cascade = InformationCascade::new(0.5, 0.9, 0.5);
        let signals = vec![PrivateSignal::Good; 10];
        let result = cascade.simulate(&signals);

        assert!(
            result
                .decisions
                .iter()
                .all(|d| d.action == CascadeAction::Adopt)
        );
    }

    #[test]
    fn cascade_all_bad_signals() {
        let cascade = InformationCascade::new(0.5, 0.9, 0.5);
        let signals = vec![PrivateSignal::Bad; 10];
        let result = cascade.simulate(&signals);

        assert!(
            result
                .decisions
                .iter()
                .all(|d| d.action == CascadeAction::Reject)
        );
    }

    #[test]
    fn cascade_triggers_with_strong_history() {
        // Strong adopt history should trigger cascade even for agents with bad signal
        let cascade = InformationCascade::new(0.5, 0.6, 0.5);
        let mut signals = vec![PrivateSignal::Good; 5]; // 5 good signals
        signals.push(PrivateSignal::Bad); // 6th agent gets bad signal
        let result = cascade.simulate(&signals);

        // Agent 5 should be in cascade and still adopt despite bad signal
        let agent5 = &result.decisions[5];
        assert!(agent5.in_cascade);
        assert_eq!(agent5.action, CascadeAction::Adopt);
    }

    #[test]
    fn cascade_start_detected() {
        let cascade = InformationCascade::new(0.5, 0.7, 0.5);
        let signals = vec![PrivateSignal::Good; 10];
        let result = cascade.simulate(&signals);

        // Cascade should start at some point
        assert!(result.cascade_start.is_some());
        assert!(result.cascade_start.unwrap() < 10);
    }

    #[test]
    fn cascade_adopt_rate() {
        let cascade = InformationCascade::new(0.5, 0.9, 0.5);
        let signals = vec![PrivateSignal::Good; 4];
        let result = cascade.simulate(&signals);
        assert!((result.adopt_rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cascade_mixed_signals_no_cascade_early() {
        let cascade = InformationCascade::new(0.5, 0.55, 0.5);
        let signals = vec![
            PrivateSignal::Good,
            PrivateSignal::Bad,
            PrivateSignal::Good,
            PrivateSignal::Bad,
        ];
        let result = cascade.simulate(&signals);
        // With low accuracy and alternating signals, first agents may not cascade
        assert!(!result.decisions[0].in_cascade);
    }

    #[test]
    fn cascade_count_matches() {
        let cascade = InformationCascade::new(0.5, 0.8, 0.5);
        let signals = vec![PrivateSignal::Good; 10];
        let result = cascade.simulate(&signals);

        let expected_count = result.decisions.iter().filter(|d| d.in_cascade).count();
        assert_eq!(result.cascade_count, expected_count);
    }

    #[test]
    fn cascade_wrong_cascade_is_possible() {
        // A cascade can form around the wrong action!
        // Strong early bad signals with moderate accuracy can cascade to reject
        // even if the true state is good.
        let cascade = InformationCascade::new(0.5, 0.65, 0.5);
        let signals = vec![PrivateSignal::Bad; 8];
        let result = cascade.simulate(&signals);

        // All reject — this could be a "wrong" cascade
        assert!(
            result
                .decisions
                .iter()
                .all(|d| d.action == CascadeAction::Reject)
        );
    }

    #[test]
    fn cascade_posterior_values() {
        let cascade = InformationCascade::new(0.5, 0.9, 0.5);
        let signals = vec![PrivateSignal::Good];
        let result = cascade.simulate(&signals);
        // P(good | good signal) = 0.9 * 0.5 / (0.9*0.5 + 0.1*0.5) = 0.9
        assert!((result.decisions[0].posterior - 0.9).abs() < 1e-10);
    }
}
