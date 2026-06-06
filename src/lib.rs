//! # Signaling Games
//!
//! Bayesian signaling games: sender/receiver equilibria, information cascades,
//! and belief revision with no external dependencies beyond `serde`.
//!
//! ## Module Overview
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`signal`] | Messages from sender, cost depends on type |
//! | [`receiver`] | Observes signal, forms posterior, chooses action |
//! | [`type_space`] | Sender's private type, prior distribution |
//! | [`equilibrium`] | Perfect Bayesian Equilibrium finder |
//! | [`cascade`] | Sequential decision model with information cascades |
//! | [`bayes`] | Bayesian prior → posterior updates |

pub mod bayes;
pub mod cascade;
pub mod equilibrium;
pub mod receiver;
pub mod signal;
pub mod type_space;
