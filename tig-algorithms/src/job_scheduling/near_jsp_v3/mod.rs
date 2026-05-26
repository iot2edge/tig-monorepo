pub mod fjsp_high;
pub mod fjsp_medium;
pub mod hybrid_flow_shop;
pub mod infra_shared;
pub mod preprocess;
pub mod types;

// job_eight-derived modules (je_ prefix). Used to upgrade the fjsp_high path
// since job_eight beats v4-derived fjsp_high by ~1,420 quality on chain.
pub mod je_fjsp_high;
pub mod je_infra;
pub mod je_preprocess;
pub mod je_types;

pub mod fs_fjsp_high;
pub mod fs_fjsp_medium;
pub mod fs_flow_shop;
pub mod fs_infra_shared;
pub mod fs_preprocess;
pub mod fs_types;

pub mod js_construction;
pub mod js_detect;
pub mod js_greedy;
pub mod js_helpers;
pub mod js_learning;
pub mod js_local_search;
pub mod js_preprocessing;
pub mod js_rules;
pub mod js_scoring;
pub mod js_track_random;
pub mod js_track_strict;
pub mod js_types;

pub mod verifier_baseline;

pub mod solver;

pub use solver::{help, solve_challenge};
