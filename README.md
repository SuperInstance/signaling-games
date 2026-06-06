# signaling-games

**Bayesian signaling games in pure Rust — sender/receiver equilibria, information cascades, and belief revision.**

Why does a peacock drag a heavy, useless tail? Why do students spend years on degrees that teach nothing directly job-relevant? Why do restaurants put expensive wine on the menu that nobody orders?

The answer: **signaling**. When one party has private information and another must decide, costly actions become credible signals. The peacock's tail says *"I'm so fit I can afford this handicap."* The degree says *"I'm competent enough to survive this gauntlet."* The expensive wine says *"this restaurant is serious about quality."*

This crate provides a complete, zero-dependency (except `serde`) framework for modeling, analyzing, and simulating **Bayesian signaling games** — the workhorse model of economics, political science, and evolutionary biology for understanding communication under asymmetric information.

---

## What This Crate Solves

You're studying a game-theoretic model where:

- A **sender** has private information (their "type") that a **receiver** can't observe directly
- The sender takes a costly, observable action (a **signal**)
- The receiver watches the signal, updates beliefs via **Bayes' rule**, and responds with an **action**
- Both parties are strategic — each maximizes their own expected payoff

This is the canonical **signaling game** (Spence, 1973). This crate lets you:

1. **Define the game** — types, signals, costs, payoffs, actions
2. **Find equilibria** — pooling, separating, and semi-separating Perfect Bayesian Equilibria
3. **Simulate cascades** — sequential agents who observe predecessors and may override private signals
4. **Do Bayesian inference** — prior → posterior updates, sequential learning, likelihood ratios

---

## The Metaphor

```
                    HIDDEN INFORMATION
                    ┌─────────────┐
                    │ Sender Type  │
                    │ (private)    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Signal Cost  │◄── Cost depends on type
                    │  (observable) │    High type → cheap signal
                    └──────┬───────┘    Low type → expensive signal
                           │
                    ┌──────▼───────┐
                    │    Signal     │──── observed by receiver
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Bayes' Rule  │──── posterior over types
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Receiver   │
                    │ Best Response│──── chooses action
                    └──────────────┘
```

Think of it as: **agents signaling hidden qualities**. The sender *knows* something the receiver doesn't. The signal is the bridge — but only if it's costly enough that low types can't cheaply mimic high types.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     signaling-games                          │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐                 │
│  │  signal   │  │type_space│  │  receiver  │                 │
│  │          │  │          │  │            │                 │
│  │ Signal   │  │SenderType│  │ Action     │                 │
│  │SignalSpace│  │TypeSpace │  │ Receiver   │                 │
│  └────┬─────┘  └────┬─────┘  │ReceiverPayoff│                │
│       │             │        └─────┬──────┘                 │
│       │    ┌────────┴──────────────┤                        │
│       ▼    ▼                       ▼                        │
│  ┌─────────────────────────────────────────┐                │
│  │              equilibrium                │                │
│  │                                         │                │
│  │  EquilibriumFinder ───► Equilibrium     │                │
│  │  • find_equilibria()                    │                │
│  │  • find_pooling() / find_separating()   │                │
│  │  • classify_strategy()                  │                │
│  │  • compute_beliefs()                    │                │
│  └──────────────┬──────────────────────────┘                │
│                 │ uses                                        │
│  ┌──────────────▼──────────────┐                             │
│  │            bayes            │                             │
│  │                             │                             │
│  │  BayesianUpdate             │                             │
│  │  • update()                 │                             │
│  │  • sequential_update()      │                             │
│  │  • marginal_likelihood()    │                             │
│  │  • log_likelihood_ratio()   │                             │
│  └──────────────┬──────────────┘                             │
│                 │ used by                                     │
│  ┌──────────────▼──────────────┐                             │
│  │           cascade           │                             │
│  │                             │                             │
│  │  InformationCascade         │                             │
│  │  PrivateSignal              │                             │
│  │  CascadeResult              │                             │
│  │  • simulate()               │                             │
│  └─────────────────────────────┘                             │
└──────────────────────────────────────────────────────────────┘
```

---

## Module Reference

| Module | Key Types | Purpose |
|--------|-----------|---------|
| `signal` | `Signal`, `SignalSpace` | Observable messages from sender; cost depends on type |
| `type_space` | `SenderType`, `TypeSpace` | Sender's private type + prior distribution |
| `receiver` | `Action`, `Receiver`, `ReceiverPayoff` | Observes signal, forms posterior, chooses action |
| `bayes` | `BayesianUpdate` | Prior → posterior via Bayes' rule; sequential updates |
| `equilibrium` | `EquilibriumFinder`, `Equilibrium`, `EquilibriumKind` | Find pooling/separating/semi-separating PBE |
| `cascade` | `InformationCascade`, `CascadeResult`, `PrivateSignal` | Sequential agents, cascade formation |

---

## Quick Start

Add to `Cargo.toml`:

```toml
[dependencies]
signaling-games = "0.1"
```

### Spence Job Market Signaling

The classic model: high-ability workers get cheap education, low-ability workers find it expensive. Employers observe education and decide whether to hire as a manager or worker.

```rust
use signaling_games::equilibrium::EquilibriumFinder;
use signaling_games::receiver::{Action, Receiver, ReceiverPayoff};
use signaling_games::signal::{Signal, SignalSpace};
use signaling_games::type_space::{SenderType, TypeSpace};

// Two worker types: high ability (cheap education) and low ability (expensive)
let type_space = TypeSpace::new(
    vec![
        SenderType::new("high".into(), 0.5),  // cost factor 0.5
        SenderType::new("low".into(), 1.5),   // cost factor 1.5
    ],
    vec![0.5, 0.5],  // equal prior
);

let signal_space = SignalSpace::new(vec![
    Signal::new("no_education".into(), 0.0),
    Signal::new("education".into(), 4.0),
]);

let receiver = Receiver::new(
    vec![
        Action::new("manager".into()),
        Action::new("worker".into()),
    ],
    ReceiverPayoff::new(vec![
        vec![10.0, 2.0],  // high type: employer prefers manager
        vec![2.0, 6.0],   // low type: employer prefers worker
    ]),
);

// Both types prefer being a manager
let sender_payoff = vec![
    vec![10.0, 2.0],  // high type payoff from (manager, worker)
    vec![8.0, 4.0],   // low type payoff from (manager, worker)
];

let finder = EquilibriumFinder::new(type_space, signal_space, receiver, sender_payoff);

// Find all Perfect Bayesian Equilibria
let equilibria = finder.find_equilibria(1e-6);
for eq in &equilibria {
    println!(
        "{:?}: high→signal[{}], low→signal[{}]",
        eq.kind, eq.sender_strategy[0], eq.sender_strategy[1]
    );
}

// Find only separating equilibria (each type sends a distinct signal)
let separating = finder.find_separating(1e-6);
assert!(!separating.is_empty());
```

### Information Cascade

Agents decide sequentially. Each sees what predecessors did, gets a private signal, and chooses whether to adopt or reject. When public evidence overwhelms private information, a **cascade** forms — agents ignore their own signal.

```rust
use signaling_games::cascade::{InformationCascade, PrivateSignal};

let cascade = InformationCascade::new(
    0.5,   // prior P(good state) = 50%
    0.8,   // signal accuracy = 80%
    0.5,   // adopt threshold = 50%
);

let signals = vec![
    PrivateSignal::Good,
    PrivateSignal::Good,
    PrivateSignal::Good,
    PrivateSignal::Bad,  // <- private signal says "bad"...
    PrivateSignal::Bad,
];

let result = cascade.simulate(&signals);

for decision in &result.decisions {
    println!(
        "Agent {}: signal={:?}, action={:?}, in_cascade={}",
        decision.agent_index,
        decision.private_signal,
        decision.action,
        decision.in_cascade,
    );
}

// Agent 3 gets a BAD signal but ADOPTS anyway — cascade has taken over
assert!(result.cascade_start.is_some());
```

### Bayesian Updating

```rust
use signaling_games::bayes::BayesianUpdate;

// Prior: 50/50 between two hypotheses
let prior = vec![0.5, 0.5];

// Evidence is 3x more likely under hypothesis 0
let posterior = BayesianUpdate::update(&prior, &[3.0, 1.0]);
assert!((posterior[0] - 0.75).abs() < 1e-10);

// Sequential updating: multiple observations
let posterior = BayesianUpdate::sequential_update(
    &[0.5, 0.5],
    &vec![vec![0.7, 0.3]; 10],  // 10 observations favoring type 0
);
// After consistent evidence, posterior strongly favors type 0
assert!(posterior[0] > 0.99);
```

---

## Mathematical Foundations

### Bayesian Updating

Given a prior distribution P(θ) over types θ and a likelihood P(e|θ) for observed evidence e:

```
P(θ|e) = P(e|θ) · P(θ) / P(e)

where P(e) = Σ_θ P(e|θ) · P(θ)
```

The `BayesianUpdate::update` function computes this exactly:

```rust
let posterior = BayesianUpdate::update(&prior, &likelihood);
// posterior[i] = prior[i] * likelihood[i] / Σ_j(prior[j] * likelihood[j])
```

### Perfect Bayesian Equilibrium (PBE)

A PBE is a triple (σ*, ρ*, μ) where:

- **σ\*(θ)**: sender strategy — maps each type to a signal
- **ρ\*(m)**: receiver strategy — maps each signal to an action
- **μ(θ|m)**: belief system — posterior probability over types given signal

Optimality conditions:

1. **Sender optimality**: For each type θ, σ\*(θ) maximizes the sender's expected utility given ρ\*
2. **Receiver optimality**: For each signal m, ρ\*(m) maximizes expected utility given μ(·|m)
3. **Belief consistency**: μ is derived from σ\* via Bayes' rule (on-path)

### Equilibrium Types

**Pooling equilibrium**: All types send the same signal. The receiver learns nothing — posterior equals prior.

```
σ(high) = σ(low) = "education"
μ(high | "education") = P(high) = 0.5
```

**Separating equilibrium**: Each type sends a distinct signal. The receiver learns the type perfectly.

```
σ(high) = "education", σ(low) = "no education"
μ(high | "education") = 1.0
μ(low  | "no education") = 1.0
```

**Semi-separating**: Some types randomize over signals.

### The Spence Model

Michael Spence's (1973) job market model:

- Two types: high ability (H) and low ability (L)
- Signal: education level e
- Cost: c_H(e) < c_L(e) — education is cheaper for high-ability workers
- Receiver: employer chooses wage w

A separating equilibrium exists when the **single-crossing condition** holds:

```
c_H(e*) - c_H(0) < w_manager - w_worker < c_L(e*) - c_L(0)
```

Education level e* is high enough that low types won't mimic it, but cheap enough that high types will.

### Information Cascades

In Banerjee's (1992) and Bikhchandani et al.'s (1992) model:

1. Agents decide sequentially: agent i observes actions a₁, ..., aᵢ₋₁
2. Agent i also receives private signal sᵢ ∈ {Good, Bad}
3. Agent i chooses aᵢ to maximize expected utility given (a₁,...,aᵢ₋₁, sᵢ)

A **cascade** occurs at agent i when:

```
argmax_action E[u(a) | a₁,...,aᵢ₋₁, sᵢ] = argmax_action E[u(a) | a₁,...,aᵢ₋₁, sᵢ']
```

for all possible signals sᵢ, sᵢ'. The agent's action is the same regardless of their private information — they've been overwhelmed by social proof.

Key insight: cascades can be **wrong**. If early agents happen to get bad signals, all subsequent agents may follow the herd despite private evidence to the contrary.

---

## Design Decisions

### Why exhaustive enumeration for equilibria?

The `EquilibriumFinder::find_equilibria` method enumerates all possible pure-strategy sender profiles. For n types and m signals, this is O(mⁿ). This is intentional:

1. **Correctness**: Complete enumeration guarantees we find *all* equilibria, not just one
2. **Educational**: The user sees every equilibrium that exists, not just the first
3. **Practical**: Most signaling games in practice have 2-3 types and 2-4 signals (8 to 64 strategies)
4. **No false negatives**: Heuristic approaches can miss equilibria; exhaustive search cannot

For games with many types/signals, the user should use `find_pooling()` or `find_separating()` to filter.

### Why `serde` as the only dependency?

Signal games are often part of larger simulation pipelines. Serde serialization enables:

- Saving/loading game configurations
- Logging equilibria and cascade results
- API serialization for web services
- Reproducible experiments

### Why `f64` throughout?

Game theory is numerical. Float precision is sufficient for equilibrium computation and Bayesian updating. The epsilon parameter in `find_equilibria` handles floating-point comparison.

### Why not generic over the action/signal types?

Concreteness over abstraction. Users define signals and actions as named entities (strings). The game-theoretic structure (payoff matrices, strategy profiles) is numerical. Mixing generics would add complexity without practical benefit.

---

## Examples

### Three-Type Signaling Game

```rust
use signaling_games::equilibrium::EquilibriumFinder;
use signaling_games::receiver::{Action, Receiver, ReceiverPayoff};
use signaling_games::signal::{Signal, SignalSpace};
use signaling_games::type_space::{SenderType, TypeSpace};

let type_space = TypeSpace::new(
    vec![
        SenderType::new("excellent".into(), 0.3),
        SenderType::new("good".into(), 0.8),
        SenderType::new("poor".into(), 2.0),
    ],
    vec![0.2, 0.5, 0.3],
);

let signal_space = SignalSpace::new(vec![
    Signal::new("no_degree".into(), 0.0),
    Signal::new("bachelors".into(), 3.0),
    Signal::new("phd".into(), 6.0),
]);

let receiver = Receiver::new(
    vec![
        Action::new("executive".into()),
        Action::new("manager".into()),
        Action::new("clerk".into()),
    ],
    ReceiverPayoff::new(vec![
        vec![10.0, 5.0, 0.0],   // excellent → executive
        vec![3.0, 8.0, 2.0],    // good → manager
        vec![0.0, 2.0, 7.0],    // poor → clerk
    ]),
);

let sender_payoff = vec![
    vec![10.0, 5.0, 0.0],
    vec![8.0, 6.0, 1.0],
    vec![6.0, 4.0, 2.0],
];

let finder = EquilibriumFinder::new(type_space, signal_space, receiver, sender_payoff);

// 3 types × 3 signals = 27 possible strategies
for eq in finder.find_equilibria(1e-6) {
    println!("{:?} equilibrium:", eq.kind);
    for (t, &s) in eq.sender_strategy.iter().enumerate() {
        println!("  type {} → signal {}", t, s);
    }
}
```

### Cascade with Variable Accuracy

```rust
use signaling_games::cascade::{InformationCascade, PrivateSignal};

// Low accuracy → slower cascade formation
let low_acc = InformationCascade::new(0.5, 0.55, 0.5);
let high_acc = InformationCascade::new(0.5, 0.95, 0.5);

let signals = vec![PrivateSignal::Good; 10];

let slow = low_acc.simulate(&signals);
let fast = high_acc.simulate(&signals);

// Higher accuracy cascades earlier
assert!(fast.cascade_start.unwrap() <= slow.cascade_start.unwrap());
println!("Low accuracy cascade starts at: {:?}", slow.cascade_start);
println!("High accuracy cascade starts at: {:?}", fast.cascade_start);
```

---

## API Stability

This is v0.1.x — the API may evolve. The core types (`Signal`, `TypeSpace`, `Receiver`, `Equilibrium`, `InformationCascade`, `BayesianUpdate`) are stable. Method signatures may gain optional parameters in future versions.

## License

MIT OR Apache-2.0

## References

- Spence, M. (1973). "Job Market Signaling." *Quarterly Journal of Economics*, 87(3), 355-374.
- Banerjee, A. (1992). "A Simple Model of Herd Behavior." *Quarterly Journal of Economics*, 107(3), 797-817.
- Bikhchandani, S., Hirshleifer, D., & Welch, I. (1992). "A Theory of Fads, Fashion, Custom, and Cultural Change as Information Cascades." *Journal of Political Economy*, 100(5), 992-1026.
- Fudenberg, D. & Tirole, J. (1991). *Game Theory*. MIT Press.
- Osborne, M. & Rubinstein, A. (1994). *A Course in Game Theory*. MIT Press.
