// Analysis tool: generate fjsp_medium instances for given (seed_string, nonce)
// pairs and dump structural features, to find what distinguishes instances
// where our solver wins vs where job_nine wins.

use tig_challenges::job_scheduling::{Challenge, Track};

/// Replicate BenchmarkSettings::calc_seed for our test setup.
/// The hashed string is: "{sorted_json}_{rand_hash}_{nonce}".
fn calc_seed(rand_hash: &str, nonce: u64) -> [u8; 32] {
    // BenchmarkSettings jsonified with sorted keys. challenge_id is c007,
    // track is the fjsp_medium track. All other id fields empty (matches
    // how scripts/test_algorithm invokes tig-runtime).
    let settings_json = r#"{"algorithm_id":"","block_id":"","challenge_id":"c007","player_id":"","track_id":"n=50,s=fjsp_medium"}"#;
    let input = format!("{}_{}_{}", settings_json, rand_hash, nonce);
    blake3::hash(input.as_bytes()).into()
}

fn make_track() -> Track {
    // Track deserializes from a "key=value,key=value" string.
    serde_json::from_value(serde_json::Value::String("n=50,s=fjsp_medium".to_string()))
        .expect("track parse")
}

struct Features {
    n_products: usize,
    n_distinct_routes: usize,
    total_ops: usize,        // sum over jobs of route length
    avg_route_len: f64,
    max_route_len: usize,
    reentrant_ops: usize,    // ops that repeat within their route
    avg_flexibility: f64,    // avg eligible machines per operation-instance
    min_flexibility: usize,
    proc_min: u32,
    proc_max: u32,
    proc_mean: f64,
    proc_std: f64,
    // total work content: sum over all job-operations of the MEAN proc time
    // of that operation across its eligible machines.
    total_work_mean: f64,
    // lower-bound-ish: total_work_mean / num_machines
    work_per_machine: f64,
    // total work using the FASTEST machine for each op (best-case)
    total_work_min: f64,
    // machine contention: for each machine, sum of min-proc-times of ops that
    // CAN run on it; report the max (the most-contended machine) / mean ratio.
    machine_load_imbalance: f64,
}

fn analyze(ch: &Challenge) -> Features {
    let n_products = ch.jobs_per_product.len();

    // Distinct routes: a "route" is the sequence of op-counts... actually we
    // only have product_processing_times (per product, the op sequence with
    // machine maps). Treat the route signature as the vector of (op index ->
    // sorted eligible machine list lengths). Two products share a route iff
    // their processing-time structures match in length & shape.
    let mut route_sigs: Vec<Vec<usize>> = Vec::new();
    for prod in &ch.product_processing_times {
        let sig: Vec<usize> = prod.iter().map(|m| m.len()).collect();
        if !route_sigs.iter().any(|r| r == &sig) {
            route_sigs.push(sig);
        }
    }
    let n_distinct_routes = route_sigs.len();

    let mut total_ops = 0usize;
    let mut route_len_sum = 0usize;
    let mut max_route_len = 0usize;
    let mut reentrant_ops = 0usize;
    let mut flex_sum = 0usize;
    let mut flex_count = 0usize;
    let mut min_flex = usize::MAX;
    let mut proc_vals: Vec<u32> = Vec::new();
    let mut total_work_mean = 0.0f64;
    let mut total_work_min = 0.0f64;
    // machine -> sum of min proc time of ops eligible on it
    let mut machine_load = vec![0.0f64; ch.num_machines];

    for (p_idx, prod) in ch.product_processing_times.iter().enumerate() {
        let jobs_in_prod = ch.jobs_per_product[p_idx];
        let route_len = prod.len();
        route_len_sum += route_len * jobs_in_prod;
        total_ops += route_len * jobs_in_prod;
        if route_len > max_route_len {
            max_route_len = route_len;
        }

        // reentrance detection: within this route, count operation steps whose
        // eligible-machine signature already appeared (proxy: identical
        // (len, sorted-machine-keys)). A cleaner detection would need op IDs,
        // which we don't have post-generation; use the machine-set as proxy.
        let mut seen_sigs: Vec<Vec<usize>> = Vec::new();
        for op_map in prod.iter() {
            let mut keys: Vec<usize> = op_map.keys().copied().collect();
            keys.sort_unstable();
            if seen_sigs.iter().any(|s| s == &keys) {
                reentrant_ops += jobs_in_prod;
            }
            seen_sigs.push(keys);

            let flex = op_map.len();
            flex_sum += flex * jobs_in_prod;
            flex_count += jobs_in_prod;
            if flex < min_flex {
                min_flex = flex;
            }

            let times: Vec<u32> = op_map.values().copied().collect();
            for &t in &times {
                proc_vals.push(t);
            }
            let mean_t = times.iter().map(|&t| t as f64).sum::<f64>() / times.len() as f64;
            let min_t = *times.iter().min().unwrap() as f64;
            total_work_mean += mean_t * jobs_in_prod as f64;
            total_work_min += min_t * jobs_in_prod as f64;

            // machine load: add min proc time to each eligible machine
            for (&m, &t) in op_map.iter() {
                machine_load[m] += (t as f64) * jobs_in_prod as f64;
            }
        }
    }

    let proc_min = *proc_vals.iter().min().unwrap();
    let proc_max = *proc_vals.iter().max().unwrap();
    let proc_mean = proc_vals.iter().map(|&t| t as f64).sum::<f64>() / proc_vals.len() as f64;
    let proc_var = proc_vals
        .iter()
        .map(|&t| {
            let d = t as f64 - proc_mean;
            d * d
        })
        .sum::<f64>()
        / proc_vals.len() as f64;
    let proc_std = proc_var.sqrt();

    let machine_load_mean = machine_load.iter().sum::<f64>() / machine_load.len() as f64;
    let machine_load_max = machine_load.iter().cloned().fold(0.0f64, f64::max);
    let machine_load_imbalance = if machine_load_mean > 0.0 {
        machine_load_max / machine_load_mean
    } else {
        0.0
    };

    Features {
        n_products,
        n_distinct_routes,
        total_ops,
        avg_route_len: route_len_sum as f64 / ch.num_jobs as f64,
        max_route_len,
        reentrant_ops,
        avg_flexibility: flex_sum as f64 / flex_count as f64,
        min_flexibility: min_flex,
        proc_min,
        proc_max,
        proc_mean,
        proc_std,
        total_work_mean,
        work_per_machine: total_work_mean / ch.num_machines as f64,
        total_work_min,
        machine_load_imbalance,
    }
}

fn main() {
    let seeds = ["jsp_compare_2026_05_13", "jsp_v4_robust_seed_2"];
    let track = make_track();

    println!(
        "{:<24} {:>5} {:>5} {:>5} {:>7} {:>7} {:>7} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>7} {:>10} {:>10} {:>7}",
        "seed", "nonce", "nProd", "nRte", "totOps", "avgRL", "maxRL", "reent",
        "avgFlx", "minFlx", "pMin", "pMax", "pMean", "pStd", "workMean", "workMin", "wPerM"
    );
    for seed in &seeds {
        for nonce in 0..10u64 {
            let s = calc_seed(seed, nonce);
            let ch = Challenge::generate_instance(&s, &track).expect("gen");
            let f = analyze(&ch);
            println!(
                "{:<24} {:>5} {:>5} {:>5} {:>7} {:>7.2} {:>7} {:>6} {:>6.2} {:>6} {:>6} {:>8} {:>8.1} {:>7.1} {:>10.0} {:>10.0} {:>7.0}",
                seed,
                nonce,
                f.n_products,
                f.n_distinct_routes,
                f.total_ops,
                f.avg_route_len,
                f.max_route_len,
                f.reentrant_ops,
                f.avg_flexibility,
                f.min_flexibility,
                f.proc_min,
                f.proc_max,
                f.proc_mean,
                f.proc_std,
                f.total_work_mean,
                f.total_work_min,
                f.work_per_machine,
            );
        }
        // per-seed averages
        println!();
    }
    let _ = machine_imbalance_note();
}

fn machine_imbalance_note() -> f64 {
    0.0
}
