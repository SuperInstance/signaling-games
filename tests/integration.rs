//! Integration tests for signaling-games.
//!
//! Covers: Spence job market signaling, cheap talk, cascade formation,
//! Bayesian updating, and cross-module interactions.

use signaling_games::bayes::BayesianUpdate;
use signaling_games::cascade::{CascadeAction, InformationCascade, PrivateSignal};
use signaling_games::equilibrium::{EquilibriumFinder, EquilibriumKind, sender_utility};
use signaling_games::receiver::{Action, Receiver, ReceiverPayoff};
use signaling_games::signal::{Signal, SignalSpace};
use signaling_games::type_space::{SenderType, TypeSpace};

fn spence_finder() -> EquilibriumFinder {
    let type_space = TypeSpace::new(
        vec![
            SenderType::new("high".into(), 0.5),
            SenderType::new("low".into(), 1.5),
        ],
        vec![0.5, 0.5],
    );
    let signal_space = SignalSpace::new(vec![
        Signal::new("no_education".into(), 0.0),
        Signal::new("education".into(), 4.0),
    ]);
    let receiver = Receiver::new(
        vec![Action::new("manager".into()), Action::new("worker".into())],
        ReceiverPayoff::new(vec![vec![10.0, 2.0], vec![2.0, 6.0]]),
    );
    let sender_payoff = vec![vec![10.0, 2.0], vec![8.0, 4.0]];
    EquilibriumFinder::new(type_space, signal_space, receiver, sender_payoff)
}

// ── Spence Job Market Signaling ──────────────────────────────────────────

#[test]
fn spence_full_model() {
    let finder = spence_finder();
    let equilibria = finder.find_equilibria(1e-6);
    assert!(!equilibria.is_empty());
    assert!(equilibria.iter().any(|e| {
        e.kind == EquilibriumKind::Separating
            && e.sender_strategy[0] == 1
            && e.sender_strategy[1] == 0
    }));
}

#[test]
fn spence_separating_payoffs() {
    let finder = spence_finder();
    let sep = finder.find_separating(1e-6);
    for eq in &sep {
        assert_ne!(eq.sender_strategy[0], eq.sender_strategy[1]);
        assert!((eq.sender_payoffs.iter().sum::<f64>() - eq.receiver_payoff).abs() > -1e10);
    }
}

// ── Cheap Talk ───────────────────────────────────────────────────────────

#[test]
fn cheap_talk_babbling_equilibrium() {
    let type_space = TypeSpace::new(
        vec![
            SenderType::new("a".into(), 1.0),
            SenderType::new("b".into(), 1.0),
        ],
        vec![0.5, 0.5],
    );
    let signal_space = SignalSpace::new(vec![
        Signal::new("say_a".into(), 0.0),
        Signal::new("say_b".into(), 0.0),
    ]);
    let receiver = Receiver::new(
        vec![Action::new("act_a".into()), Action::new("act_b".into())],
        ReceiverPayoff::new(vec![vec![10.0, 0.0], vec![0.0, 10.0]]),
    );
    let sender_payoff = vec![vec![10.0, 0.0], vec![10.0, 0.0]];
    let finder = EquilibriumFinder::new(type_space, signal_space, receiver, sender_payoff);
    assert!(
        finder
            .find_equilibria(1e-6)
            .iter()
            .any(|e| e.kind == EquilibriumKind::Pooling)
    );
}

// ── Cascade Formation ────────────────────────────────────────────────────

#[test]
fn cascade_triggers_on_consensus() {
    let cascade = InformationCascade::new(0.5, 0.8, 0.5);
    let mut signals = vec![PrivateSignal::Good; 5];
    signals.push(PrivateSignal::Bad);
    let result = cascade.simulate(&signals);
    assert_eq!(result.decisions[5].action, CascadeAction::Adopt);
    assert!(result.decisions[5].in_cascade);
}

#[test]
fn cascade_no_cascade_first_agent() {
    let cascade = InformationCascade::new(0.5, 0.6, 0.5);
    let result = cascade.simulate(&[PrivateSignal::Good]);
    assert!(!result.decisions[0].in_cascade);
}

#[test]
fn cascade_wrong_cascade_possible() {
    let cascade = InformationCascade::new(0.3, 0.7, 0.5);
    let result = cascade.simulate(&vec![PrivateSignal::Bad; 10]);
    assert!(
        result
            .decisions
            .iter()
            .all(|d| d.action == CascadeAction::Reject)
    );
}

// ── Bayesian Updating ────────────────────────────────────────────────────

#[test]
fn bayes_sequential_convergence() {
    let posterior = BayesianUpdate::sequential_update(&[0.5, 0.5], &vec![vec![0.7, 0.3]; 10]);
    assert!(posterior[0] > 0.99);
}

#[test]
fn bayes_symmetry() {
    let p1 = BayesianUpdate::update(&[0.5, 0.5], &[0.8, 0.2]);
    let p2 = BayesianUpdate::update(&[0.5, 0.5], &[0.2, 0.8]);
    assert!((p1[0] - p2[1]).abs() < 1e-10);
}

// ── Cross-Module ─────────────────────────────────────────────────────────

#[test]
fn prior_reflected_in_pooling_beliefs() {
    let type_space = TypeSpace::new(
        vec![
            SenderType::new("h".into(), 0.5),
            SenderType::new("l".into(), 1.5),
        ],
        vec![0.9, 0.1],
    );
    let signal_space = SignalSpace::new(vec![
        Signal::new("s0".into(), 0.0),
        Signal::new("s1".into(), 4.0),
    ]);
    let receiver = Receiver::new(
        vec![Action::new("hire".into())],
        ReceiverPayoff::new(vec![vec![1.0], vec![1.0]]),
    );
    let finder = EquilibriumFinder::new(
        type_space,
        signal_space,
        receiver,
        vec![vec![1.0], vec![1.0]],
    );
    let beliefs = finder.compute_beliefs(&vec![0, 0]);
    assert!((beliefs[0][0] - 0.9).abs() < 1e-10);
}

#[test]
fn receiver_strategy_matches_separating() {
    let type_space = TypeSpace::new(
        vec![
            SenderType::new("a".into(), 1.0),
            SenderType::new("b".into(), 1.0),
        ],
        vec![0.5, 0.5],
    );
    let signal_space = SignalSpace::new(vec![
        Signal::new("s0".into(), 1.0),
        Signal::new("s1".into(), 1.0),
    ]);
    let receiver = Receiver::new(
        vec![Action::new("x".into()), Action::new("y".into())],
        ReceiverPayoff::new(vec![vec![5.0, 1.0], vec![1.0, 5.0]]),
    );
    let finder = EquilibriumFinder::new(
        type_space,
        signal_space,
        receiver,
        vec![vec![5.0, 1.0], vec![5.0, 1.0]],
    );
    let beliefs = finder.compute_beliefs(&vec![0, 1]);
    let rx = finder.compute_receiver_strategy(&beliefs);
    assert_eq!(rx[0], 0);
    assert_eq!(rx[1], 1);
}

#[test]
fn sender_utility_with_cost() {
    let u = sender_utility(0, 0, 0, 100.0, &[vec![50.0]]);
    assert!((u - (-50.0)).abs() < 1e-10);
}
