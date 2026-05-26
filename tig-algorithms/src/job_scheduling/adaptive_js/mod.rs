pub mod construction;
pub mod detect;
pub mod greedy;
pub mod helpers;
pub mod learning;
pub mod local_search;
pub mod preprocessing;
pub mod rules;
pub mod scoring;
pub mod solver;
pub mod track_chaotic;
pub mod track_complex;
pub mod track_parallel;
pub mod track_random;
pub mod track_strict;
pub mod types;

pub use solver::{help, solve_challenge};
