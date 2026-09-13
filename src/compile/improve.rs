//! smysl-optimize **Phase D** — the automatic improve-loop CONTROLLER.
//!
//! `generate → rank → regenerate` with the smysl corpus as tabu memory (Phase B) and a convergence guard,
//! so prompt search *converges* instead of death-marching. The controller is PURE over an injected
//! [`ImproveStep`] (propose a delta / rank a prompt), so the novel, error-prone part — the loop logic,
//! the tabu integration, the stopping rules — is gate-tested offline. The real step (an LLM regenerator +
//! compile + render + aesthetic rank) wires in at the CLI, where it needs live models.
//!
//! The loop guarantees, by construction, what the human death march cannot: it never re-tries a move it
//! already tried (every tried delta joins the tabu list), and it always stops with a *reason* and the
//! best-so-far — on budget, on a plateau, when the proposer runs dry, or when it insists on a spent move.

use crate::smysl::tabu_reason;

/// One improve pass: the delta it tried, the rank that delta earned, and whether it was kept.
#[derive(Debug, Clone)]
pub struct Pass {
    pub n: usize,
    /// `(old, new)` verbatim replacement; `None` only for the baseline pass 0.
    pub delta: Option<(String, String)>,
    pub rank: f32,
    pub kept: bool,
}

/// Why the loop stopped — always reported, so a run never just "gives up" silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Ran the full pass budget.
    Budget,
    /// `plateau_k` passes in a row with no gain above `min_gain`.
    Plateau,
    /// The proposer had no fresh (non-tabu) move left.
    NoMoves,
    /// The proposer kept insisting on a tabu move — a stuck oscillation.
    Oscillation,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::Budget => "pass budget spent",
            StopReason::Plateau => "rank plateaued",
            StopReason::NoMoves => "no fresh moves left",
            StopReason::Oscillation => "stuck (proposer kept repeating a spent move)",
        }
    }
}

/// The result of a loop: the best prompt found, its rank, the per-pass trace, and why it stopped.
#[derive(Debug, Clone)]
pub struct ImproveOutcome {
    pub best_prompt: String,
    pub best_rank: f32,
    pub passes: Vec<Pass>,
    pub stop: StopReason,
}

impl ImproveOutcome {
    /// The tried deltas as `(old, new, why)` triples — for recording into the corpus (each becomes a tabu
    /// move next run) via [`crate::smysl::fixes_to_records`]. The `why` carries the pass + rank outcome.
    pub fn corpus_moves(&self) -> Vec<(String, String, String)> {
        self.passes
            .iter()
            .filter_map(|p| {
                p.delta.as_ref().map(|(o, n)| {
                    let why = format!(
                        "aesthetic pass {}: rank {:.2} ({})",
                        p.n,
                        p.rank,
                        if p.kept { "kept" } else { "reverted — do not retry" }
                    );
                    (o.clone(), n.clone(), why)
                })
            })
            .collect()
    }
}

/// The work each pass injects. The controller owns the tabu list, the convergence guard, and the
/// best-so-far; the step only proposes a delta and ranks a prompt.
pub trait ImproveStep {
    /// Propose one delta `(old, new)` to improve `prompt`, avoiding the `tabu` moves. `None` means the
    /// proposer has nothing fresh — the loop then stops with [`StopReason::NoMoves`].
    fn propose(&mut self, prompt: &str, tabu: &[(String, String)]) -> Option<(String, String)>;
    /// Render + rank a prompt; higher is better (e.g. the LAION aesthetic score). In production this
    /// compiles the prompt, renders it, and scores the image; here it is injected so the loop is testable.
    fn rank(&mut self, prompt: &str) -> f32;
}

/// Apply a delta to a prompt: verbatim replace of the first `old` with `new` (empty `new` deletes).
fn apply_delta(prompt: &str, old: &str, new: &str) -> String {
    prompt.replacen(old, new, 1)
}

/// Run the automatic improve loop.
///
/// `seed_tabu` is the corpus's prior moves (Phase B), so the loop starts already knowing what not to try.
/// A delta is *kept* only if it beats the best-so-far by more than `min_gain` (the noise floor — aesthetic
/// rank is noisy, so a hair of improvement is not improvement). Every tried delta joins the tabu list, so
/// the loop physically cannot re-march. Stops on budget / plateau / dry proposer / oscillation, always with
/// the best prompt found.
pub fn run_improve(
    step: &mut dyn ImproveStep,
    initial_prompt: &str,
    seed_tabu: Vec<(String, String)>,
    max_passes: usize,
    plateau_k: usize,
    min_gain: f32,
) -> ImproveOutcome {
    let mut tabu = seed_tabu;
    let mut prompt = initial_prompt.to_string();
    let base = step.rank(&prompt);
    let mut best = (prompt.clone(), base);
    let mut passes = vec![Pass { n: 0, delta: None, rank: base, kept: true }];
    let mut stale = 0usize;
    let mut oscillated = 0usize;

    for n in 1..=max_passes {
        let Some((old, new)) = step.propose(&prompt, &tabu) else {
            return finish(best, passes, StopReason::NoMoves);
        };
        // The proposer should honour the tabu; the controller GUARANTEES it (the Phase B safety net). A
        // proposer that keeps handing back spent moves is stuck — stop rather than spin.
        if tabu_reason(&old, &new, &tabu).is_some() {
            oscillated += 1;
            if oscillated >= 2 {
                return finish(best, passes, StopReason::Oscillation);
            }
            continue;
        }
        oscillated = 0;

        let candidate = apply_delta(&prompt, &old, &new);
        let r = step.rank(&candidate);
        tabu.push((old.clone(), new.clone())); // tried once → never again

        let kept = r > best.1 + min_gain;
        if kept {
            prompt = candidate;
            best = (prompt.clone(), r);
            stale = 0;
        } else {
            stale += 1;
        }
        passes.push(Pass { n, delta: Some((old, new)), rank: r, kept });

        if stale >= plateau_k {
            return finish(best, passes, StopReason::Plateau);
        }
    }
    finish(best, passes, StopReason::Budget)
}

fn finish(best: (String, f32), passes: Vec<Pass>, stop: StopReason) -> ImproveOutcome {
    ImproveOutcome { best_prompt: best.0, best_rank: best.1, passes, stop }
}

/// A human-readable summary of a loop run — the per-pass rank trajectory + why it stopped + the best.
pub fn format_report(o: &ImproveOutcome) -> String {
    let mut s = String::new();
    for p in &o.passes {
        match &p.delta {
            None => s.push_str(&format!("  pass 0 · baseline · rank {:.2}\n", p.rank)),
            Some((old, new)) => s.push_str(&format!(
                "  pass {} · rank {:.2} · {} · “{}” → “{}”\n",
                p.n,
                p.rank,
                if p.kept { "kept" } else { "reverted" },
                clip(old),
                clip(new),
            )),
        }
    }
    s.push_str(&format!(
        "  ↳ stopped: {} · best rank {:.2} after {} pass(es)\n",
        o.stop.as_str(),
        o.best_rank,
        o.passes.len().saturating_sub(1),
    ));
    s
}

fn clip(s: &str) -> String {
    let s = s.trim().replace('\n', " ");
    if s.chars().count() > 32 {
        format!("{}…", s.chars().take(31).collect::<String>())
    } else if s.is_empty() {
        "(removed)".into()
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic stub: aesthetic = count of `★` in the prompt; the proposer offers a fixed list of
    /// candidate deltas in order, skipping any that are already tabu.
    struct StubStep {
        moves: Vec<(String, String)>,
        idx: usize,
    }
    impl StubStep {
        fn new(moves: &[(&str, &str)]) -> Self {
            StubStep { moves: moves.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(), idx: 0 }
        }
    }
    impl ImproveStep for StubStep {
        fn propose(&mut self, _prompt: &str, tabu: &[(String, String)]) -> Option<(String, String)> {
            while self.idx < self.moves.len() {
                let m = self.moves[self.idx].clone();
                self.idx += 1;
                if tabu_reason(&m.0, &m.1, tabu).is_none() {
                    return Some(m);
                }
            }
            None
        }
        fn rank(&mut self, prompt: &str) -> f32 {
            prompt.matches('★').count() as f32
        }
    }

    #[test]
    fn climbs_to_the_best_and_keeps_only_improvements() {
        // Two deltas each add a ★ (rank climbs); one adds nothing (reverted).
        let mut step = StubStep::new(&[("lane", "lane ★"), ("dawn", "dawn"), ("fog", "fog ★")]);
        let out = run_improve(&mut step, "a lane at dawn in fog", vec![], 10, 3, 0.0);
        assert_eq!(out.best_rank, 2.0, "kept both ★-adding deltas");
        assert!(out.best_prompt.contains("lane ★") && out.best_prompt.contains("fog ★"));
        // The neutral delta was tried but not kept.
        assert!(out.passes.iter().any(|p| p.delta.as_ref().map(|(o, _)| o == "dawn").unwrap_or(false) && !p.kept));
    }

    #[test]
    fn stops_on_plateau_with_best_so_far() {
        // Every move is neutral (no ★) → no gain → plateau after plateau_k passes.
        let mut step = StubStep::new(&[("a", "a"), ("b", "b"), ("c", "c"), ("d", "d")]);
        let out = run_improve(&mut step, "a b c d", vec![], 10, 2, 0.0);
        assert_eq!(out.stop, StopReason::Plateau);
        assert_eq!(out.best_rank, 0.0);
    }

    #[test]
    fn never_retries_a_tabu_move() {
        // Seed the tabu with the only ★-move; the proposer offers it, the loop must skip it and run dry.
        let mut step = StubStep::new(&[("lane", "lane ★")]);
        let seed = vec![("lane".to_string(), "lane ★".to_string())];
        let out = run_improve(&mut step, "a lane", seed, 10, 3, 0.0);
        assert_eq!(out.stop, StopReason::NoMoves, "the only move was tabu → nothing fresh to try");
        assert_eq!(out.best_rank, 0.0, "never applied the spent move");
        assert!(!out.best_prompt.contains('★'));
    }

    #[test]
    fn tried_deltas_become_corpus_moves() {
        let mut step = StubStep::new(&[("lane", "lane ★")]);
        let out = run_improve(&mut step, "a lane", vec![], 10, 3, 0.0);
        let moves = out.corpus_moves();
        assert_eq!(moves.len(), 1);
        assert_eq!((moves[0].0.as_str(), moves[0].1.as_str()), ("lane", "lane ★"));
        assert!(moves[0].2.contains("aesthetic pass 1") && moves[0].2.contains("kept"));
    }

    #[test]
    fn min_gain_rejects_noise() {
        // A move that improves by exactly the noise floor is NOT kept (must beat it).
        struct Tiny;
        impl ImproveStep for Tiny {
            fn propose(&mut self, _p: &str, tabu: &[(String, String)]) -> Option<(String, String)> {
                let m = ("x".to_string(), "x+".to_string());
                if tabu_reason(&m.0, &m.1, tabu).is_none() { Some(m) } else { None }
            }
            fn rank(&mut self, prompt: &str) -> f32 {
                if prompt.contains("x+") { 5.1 } else { 5.0 }
            }
        }
        let out = run_improve(&mut Tiny, "x", vec![], 5, 2, 0.2); // gain 0.1 < min_gain 0.2
        assert!(out.passes.iter().all(|p| p.n == 0 || !p.kept), "a sub-noise gain is not an improvement");
        assert_eq!(out.best_rank, 5.0);
    }
}
