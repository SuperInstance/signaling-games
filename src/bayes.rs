//! Bayesian updating: prior → posterior given observed evidence.
//!
//! The mathematical backbone of signaling games. Given a prior distribution
//! over types and a likelihood function P(evidence | type), compute the
//! posterior distribution P(type | evidence) using Bayes' rule:
//!
//! ```text
//! P(type | evidence) = P(evidence | type) × P(type) / P(evidence)
//! ```

use serde::{Deserialize, Serialize};

/// A Bayesian updater that computes posteriors from priors and likelihoods.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BayesianUpdate;

impl BayesianUpdate {
    /// Compute posterior from prior and likelihood.
    ///
    /// - `prior`: P(type_i) for each type
    /// - `likelihood`: P(evidence | type_i) for each type
    ///
    /// Returns the posterior P(type_i | evidence) for each type.
    ///
    /// # Panics
    ///
    /// Panics if lengths don't match or all likelihoods are zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use signaling_games::bayes::BayesianUpdate;
    ///
    /// // Prior: 50/50, but evidence is twice as likely under type 0
    /// let posterior = BayesianUpdate::update(&[0.5, 0.5], &[2.0, 1.0]);
    /// assert!((posterior[0] - 2.0 / 3.0).abs() < 1e-10);
    /// assert!((posterior[1] - 1.0 / 3.0).abs() < 1e-10);
    /// ```
    pub fn update(prior: &[f64], likelihood: &[f64]) -> Vec<f64> {
        assert_eq!(
            prior.len(),
            likelihood.len(),
            "prior and likelihood must have same length"
        );

        let joint: Vec<f64> = prior
            .iter()
            .zip(likelihood.iter())
            .map(|(&p, &l)| p * l)
            .collect();

        let evidence: f64 = joint.iter().copied().sum();
        assert!(
            evidence > 0.0,
            "total evidence probability must be positive"
        );

        joint.iter().map(|&j| j / evidence).collect()
    }

    /// Compute the log-likelihood ratio between two types given evidence.
    ///
    /// Returns `ln(P(evidence | type_a) / P(evidence | type_b))`.
    pub fn log_likelihood_ratio(likelihood_a: f64, likelihood_b: f64) -> f64 {
        assert!(
            likelihood_a > 0.0 && likelihood_b > 0.0,
            "likelihoods must be positive"
        );
        (likelihood_a / likelihood_b).ln()
    }

    /// Sequential Bayesian update: update a prior with multiple observations.
    ///
    /// Each observation provides a new likelihood vector. The posterior after
    /// one observation becomes the prior for the next.
    pub fn sequential_update(prior: &[f64], likelihoods: &[Vec<f64>]) -> Vec<f64> {
        let mut current = prior.to_vec();
        for like in likelihoods {
            current = Self::update(&current, like);
        }
        current
    }

    /// Compute the marginal likelihood (evidence term).
    ///
    /// P(evidence) = Σ_i P(evidence | type_i) × P(type_i)
    pub fn marginal_likelihood(prior: &[f64], likelihood: &[f64]) -> f64 {
        prior
            .iter()
            .zip(likelihood.iter())
            .map(|(&p, &l)| p * l)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_two_type_update() {
        let posterior = BayesianUpdate::update(&[0.5, 0.5], &[0.8, 0.2]);
        assert!((posterior[0] - 0.8).abs() < 1e-10);
        assert!((posterior[1] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn uniform_prior_update() {
        // Three types, uniform prior, evidence strongly favors type 1
        let posterior =
            BayesianUpdate::update(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], &[0.1, 0.8, 0.1]);
        assert!((posterior[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn degenerate_evidence() {
        // Evidence impossible under type 0
        let posterior = BayesianUpdate::update(&[0.5, 0.5], &[0.0, 1.0]);
        assert!((posterior[0] - 0.0).abs() < 1e-10);
        assert!((posterior[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn mismatched_lengths() {
        BayesianUpdate::update(&[0.5, 0.5], &[1.0]);
    }

    #[test]
    fn log_likelihood_ratio() {
        let llr = BayesianUpdate::log_likelihood_ratio(4.0, 2.0);
        assert!((llr - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn sequential_update() {
        let prior = vec![0.5, 0.5];
        let observations = vec![vec![0.7, 0.3], vec![0.6, 0.4]];
        let posterior = BayesianUpdate::sequential_update(&prior, &observations);
        // After two updates favoring type 0, posterior[0] should be > 0.5
        assert!(posterior[0] > 0.7);
        assert!((posterior.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn marginal_likelihood_computation() {
        let ml = BayesianUpdate::marginal_likelihood(&[0.5, 0.5], &[0.8, 0.2]);
        assert!((ml - 0.5).abs() < 1e-10);
    }

    #[test]
    fn posterior_sums_to_one() {
        let posterior = BayesianUpdate::update(&[0.3, 0.3, 0.4], &[0.5, 0.3, 0.2]);
        assert!((posterior.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }
}
