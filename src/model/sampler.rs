//! Token sampling strategies: temperature, top-k, top-p (nucleus).

use std::collections::HashMap;

/// Configuration for sampling.
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,       // 0 = disabled
            top_p: 1.0,     // 1.0 = disabled
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        }
    }
}

/// Sample a token from logits using the given configuration.
///
/// Steps:
/// 1. Apply repetition penalty to tokens seen in `history`
/// 2. Apply temperature scaling
/// 3. Top-k filtering (keep only top-k logits, mask rest to -inf)
/// 4. Top-p (nucleus) filtering
/// 5. Sample from the resulting distribution
pub fn sample(logits: &[f32], cfg: &SamplerConfig, history: &[u32]) -> u32 {
    assert!(!logits.is_empty(), "logits must not be empty");

    let n_vocab = logits.len();
    let mut scored: Vec<(u32, f32)> = (0..n_vocab as u32)
        .zip(logits.iter().copied())
        .collect();

    // 1. Repetition penalty
    if cfg.repeat_penalty != 1.0 && !history.is_empty() {
        let recent_start = history.len().saturating_sub(cfg.repeat_last_n);
        let mut seen = HashMap::new();
        for &tok in &history[recent_start..] {
            *seen.entry(tok).or_insert(0) += 1;
        }
        for (id, logit) in &mut scored {
            if seen.get(id).is_some_and(|&c| c > 0) {
                if *logit > 0.0 {
                    *logit /= cfg.repeat_penalty;
                } else {
                    *logit *= cfg.repeat_penalty;
                }
            }
        }
    }

    // 2. Temperature scaling
    let temp = cfg.temperature.max(1e-6);
    if (temp - 1.0).abs() > 1e-6 {
        for (_, logit) in &mut scored {
            *logit /= temp;
        }
    }

    // 3. Top-k filtering
    let mut valid_count = scored.len();
    if cfg.top_k > 0 && cfg.top_k < scored.len() {
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(cfg.top_k);
        valid_count = scored.len();
    }

    // 4. Top-p (nucleus) filtering
    if cfg.top_p < 1.0 && valid_count > 1 {
        // Sort by descending logit
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Convert to probs for cumulative sum
        let max_logit = scored[0].1;
        let exp_sum: f64 = scored
            .iter()
            .map(|(_, l)| ((l - max_logit) as f64).exp())
            .sum();
        let mut cumprob = 0.0f64;
        let mut cutoff = valid_count;
        for (i, (_, l)) in scored.iter().enumerate() {
            let prob = ((l - max_logit) as f64).exp() / exp_sum;
            cumprob += prob;
            if cumprob > cfg.top_p as f64 {
                cutoff = i + 1;
                break;
            }
        }
        scored.truncate(cutoff.max(1));
    }

    // 5. Sample from distribution (softmax of remaining logits, then pick)
    let max_logit = scored
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);

    // Use f64 for numerical stability
    let exp_sum: f64 = scored
        .iter()
        .map(|(_, l)| ((l - max_logit) as f64).exp())
        .sum();

    // Generate a random number in [0, 1)
    let r: f64 = {
        // Simple xorshift64 PRNG seeded from a hash of the logit values
        // Not cryptographically secure, but adequate for sampling
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        for (_, l) in &scored {
            l.to_bits().hash(&mut hasher);
        }
        // Use system time for entropy
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        nanos.hash(&mut hasher);
        let h = hasher.finish();
        (h >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut cumulative = 0.0f64;
    for (id, l) in &scored {
        let prob = ((l - max_logit) as f64).exp() / exp_sum;
        cumulative += prob;
        if r < cumulative {
            return *id;
        }
    }

    // Fallback: return last token (shouldn't reach here)
    scored.last().unwrap().0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_equivalent() {
        // With temperature=1.0, no penalty, top_k=0, top_p=1.0 -> should pick highest logit
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let cfg = SamplerConfig {
            temperature: 0.01, // very low temperature -> almost deterministic
            ..Default::default()
        };
        let token = sample(&logits, &cfg, &[]);
        assert_eq!(token, 1);
    }

    #[test]
    fn temperature_raises_low_tokens() {
        // With high temperature, distribution flattens
        let logits = vec![0.0, 100.0];
        let cfg = SamplerConfig {
            temperature: 100.0,
            ..Default::default()
        };
        // Over many samples, should sometimes pick token 0
        let mut picked_zero = false;
        for _ in 0..1000 {
            if sample(&logits, &cfg, &[]) == 0 {
                picked_zero = true;
                break;
            }
        }
        assert!(picked_zero, "high temperature should sometimes pick non-argmax token");
    }

    #[test]
    fn repetition_penalty() {
        let logits = vec![5.0, 5.0];
        let history = vec![0, 0, 0];
        let cfg = SamplerConfig {
            temperature: 0.01,
            repeat_penalty: 10.0,
            ..Default::default()
        };
        // Token 0 has been seen 3 times, penalty should push it down
        let token = sample(&logits, &cfg, &history);
        assert_eq!(token, 1);
    }

    #[test]
    fn top_k_limits_choices() {
        // Top-k = 1 means always pick the best
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_k: 1,
            ..Default::default()
        };
        for _ in 0..100 {
            assert_eq!(sample(&logits, &cfg, &[]), 1);
        }
    }
}
