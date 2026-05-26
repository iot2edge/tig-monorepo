// v4-derived (top-level) modules. Only fjsp_medium remains wired (Complex
// low-fuel fallback); preprocess/types/infra_shared support it.
pub mod fjsp_medium;
pub mod infra_shared;
pub mod preprocess;
pub mod types;

// job_eight-derived modules (je_ prefix). Used for fjsp_high (proven win).
pub mod je_fjsp_high;
pub mod je_infra;
pub mod je_preprocess;
pub mod je_types;

// job_nine-derived modules (jn_ prefix). Used for flow_shop, hybrid_flow_shop,
// job_shop (high fuel), and fjsp_medium (high fuel) — job_nine wins these at 5T.
pub mod jn_fjsp_medium;
pub mod jn_flow_shop;
pub mod jn_hybrid_flow_shop;
pub mod jn_infra;
pub mod jn_job_shop;
pub mod jn_preprocess;
pub mod jn_types;

// v1-derived (js_ prefix) modules. Used for job_shop (Random, low-fuel dual).
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
pub mod js_types;

pub mod verifier_baseline;

pub mod solver;

pub use solver::{help, solve_challenge};
