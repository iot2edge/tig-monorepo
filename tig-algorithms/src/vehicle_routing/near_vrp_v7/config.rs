use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Config {
    pub exploration_level: usize,
    pub allow_swap3: bool,
    pub granularity: usize,
    pub granularity2: usize,
    pub penalty_tw: usize,
    pub penalty_capa: usize,
    pub target_ratio: f64,
    pub max_it_noimprov: usize,
    pub max_it_total: usize,
    pub nb_it_adapt_penalties: usize,
    pub nb_it_traces: usize,
    pub mu: usize,
    pub mu_start: usize,
    pub lambda: usize,
    pub nb_close: usize,
    pub nb_elite: usize,
    // SISR (Slack Induction by String Removals) ILS-style diversification kick.
    // When the HGS stagnates (it_noimprov climbing), perturb the incumbent best
    // via ruin-and-recreate, re-optimise with the existing LS, and re-admit it.
    #[serde(default)]
    pub sisr_enabled: bool,
    #[serde(default = "default_sisr_kick_interval")]
    pub sisr_kick_interval: usize,
    #[serde(default = "default_sisr_c_bar")]
    pub sisr_c_bar: f64,
    #[serde(default = "default_sisr_l_max")]
    pub sisr_l_max: usize,
    #[serde(default = "default_sisr_alpha")]
    pub sisr_alpha: f64,
    #[serde(default = "default_sisr_blink")]
    pub sisr_blink: f64,
}

fn default_sisr_kick_interval() -> usize { 25 }
fn default_sisr_c_bar() -> f64 { 10.0 }
fn default_sisr_l_max() -> usize { 10 }
fn default_sisr_alpha() -> f64 { 0.5 }
fn default_sisr_blink() -> f64 { 0.01 }

impl Config {
    fn preset(exploration_level: usize, nb_nodes: usize) -> Self {
        let p = if nb_nodes <= 700 {
            20
        } else if nb_nodes <= 1000 {
            30
        } else if nb_nodes <= 1200 {
            50
        } else if nb_nodes <= 1500 {
            80
        } else if nb_nodes <= 2000 {
            150
        } else if nb_nodes <= 3000 {
            200
        } else {
            500
        };

        match exploration_level {
            0 => Self {
                exploration_level: 0,
                allow_swap3: true,
                granularity: 40,
                granularity2: 20,
                penalty_tw: p,
                penalty_capa: p,
                target_ratio: 0.2,
                max_it_noimprov: 0,
                max_it_total: 0,
                nb_it_adapt_penalties: 100,
                nb_it_traces: 100,
                mu: 2,
                mu_start: 1,
                lambda: 1,
                nb_close: 1,
                nb_elite: 1,
                sisr_enabled: false,
                sisr_kick_interval: 25,
                sisr_c_bar: 10.0,
                sisr_l_max: 10,
                sisr_alpha: 0.5,
                sisr_blink: 0.01,
            },
            1 => Self {
                exploration_level: 0,
                allow_swap3: true,
                granularity: 40,
                granularity2: 20,
                penalty_tw: p,
                penalty_capa: p,
                target_ratio: 0.2,
                max_it_noimprov: 0,
                max_it_total: 0,
                nb_it_adapt_penalties: 100,
                nb_it_traces: 100,
                mu: 2,
                mu_start: 5,
                lambda: 1,
                nb_close: 1,
                nb_elite: 1,
                sisr_enabled: false,
                sisr_kick_interval: 25,
                sisr_c_bar: 10.0,
                sisr_l_max: 10,
                sisr_alpha: 0.5,
                sisr_blink: 0.01,
            },
            2 => Self {
                exploration_level: 1,
                allow_swap3: true,
                granularity: 40,
                granularity2: 20,
                penalty_tw: p,
                penalty_capa: p,
                target_ratio: 0.2,
                max_it_noimprov: 10,
                max_it_total: 50,
                nb_it_adapt_penalties: 100,
                nb_it_traces: 100,
                mu: 3,
                mu_start: 6,
                lambda: 3,
                nb_close: 1,
                nb_elite: 1,
                sisr_enabled: false,
                sisr_kick_interval: 25,
                sisr_c_bar: 10.0,
                sisr_l_max: 10,
                sisr_alpha: 0.5,
                sisr_blink: 0.01,
            },
            3 => Self {
                exploration_level: 2,
                allow_swap3: true,
                granularity: 40,
                granularity2: 20,
                penalty_tw: p,
                penalty_capa: p,
                target_ratio: 0.2,
                max_it_noimprov: 100,
                max_it_total: 500,
                nb_it_adapt_penalties: 20,
                nb_it_traces: 20,
                mu: 5,
                mu_start: 10,
                lambda: 5,
                nb_close: 2,
                nb_elite: 2,
                sisr_enabled: true,
                sisr_kick_interval: 25,
                sisr_c_bar: 10.0,
                sisr_l_max: 10,
                sisr_alpha: 0.5,
                sisr_blink: 0.01,
            },
            4 => Self {
                exploration_level: 3,
                allow_swap3: false,
                granularity: 30,
                granularity2: 20,
                penalty_tw: p,
                penalty_capa: p,
                target_ratio: 0.2,
                max_it_noimprov: 500,
                max_it_total: 5_000,
                nb_it_adapt_penalties: 20,
                nb_it_traces: 100,
                mu: 10,
                mu_start: 20,
                lambda: 10,
                nb_close: 2,
                nb_elite: 3,
                sisr_enabled: true,
                sisr_kick_interval: 25,
                sisr_c_bar: 10.0,
                sisr_l_max: 10,
                sisr_alpha: 0.5,
                sisr_blink: 0.01,
            },
            _ => Self::preset(4, nb_nodes),
        }
    }

    pub fn defaults(nb_nodes: usize) -> Self {
        Self::preset(0, nb_nodes)
    }

    pub fn initialize(hyperparameters: &Option<Map<String, Value>>, nb_nodes: usize) -> Self {
        let mut base_params = Self::defaults(nb_nodes);

        if let Some(v) = hyperparameters.as_ref().and_then(|m| m.get("exploration_level")) {
            match v {
                Value::Number(n) => {
                    if let Some(u) = n.as_u64() {
                        base_params = Self::preset(u as usize, nb_nodes);
                    }
                }
                Value::String(s) => {
                    if let Ok(u) = s.parse::<usize>() {
                        base_params = Self::preset(u, nb_nodes);
                    }
                }
                _ => {}
            }
        }

        let mut merged_params = serde_json::to_value(base_params).expect("Config serializable");
        if let (Value::Object(ref mut obj), Some(map)) = (&mut merged_params, hyperparameters) {
            for (k, v) in map {
                if k == "exploration_level" {
                    continue;
                }
                obj.insert(k.clone(), v.clone());
            }
        }

        serde_json::from_value(merged_params).unwrap_or_else(|_| Self::defaults(nb_nodes))
    }
}
