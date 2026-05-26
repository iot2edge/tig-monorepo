pub mod types;
pub mod preprocess;
pub mod infra_shared;
pub mod hybrid_flow_shop;
pub mod fjsp_medium;
pub mod fjsp_high;

// job_eight-derived modules (je_ prefix). Used for fjsp_high (proven win) and
// for flow_shop where job_eight beats our prior solver.
pub mod je_types;
pub mod je_infra;
pub mod je_preprocess;
pub mod je_fjsp_high;
pub mod je_flow_shop;

// job_nine-derived modules (jn_ prefix). Used for hybrid_flow_shop where
// job_nine beats both our solver and job_eight.
pub mod jn_types;
pub mod jn_infra;
pub mod jn_preprocess;
pub mod jn_our_search;
pub mod jn_job_shop;
pub mod jn_hybrid_flow_shop;

pub mod fs_types;
pub mod fs_preprocess;
pub mod fs_infra_shared;
pub mod fs_flow_shop;
pub mod fs_fjsp_medium;
pub mod fs_fjsp_high;

pub mod js_types;
pub mod js_preprocessing;
pub mod js_helpers;
pub mod js_greedy;
pub mod js_detect;
pub mod js_construction;
pub mod js_learning;
pub mod js_local_search;
pub mod js_rules;
pub mod js_scoring;
pub mod js_track_random;
pub mod js_track_strict;

pub mod verifier_baseline;

pub mod solver;

pub use solver::{solve_challenge, help};
