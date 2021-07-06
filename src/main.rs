use noisy_float::prelude::*;
use rand::prelude::*;
use rand_distr::{Exp, Pareto, WeightedAliasIndex};

use std::collections::VecDeque;
use std::f64::INFINITY;

const EPSILON: f64 = 1e-8;

#[derive(Clone, Debug)]
enum Dist {
    Pareto(Pareto<f64>, f64, f64),
    NModal(WeightedAliasIndex<f64>, Vec<f64>, Vec<f64>),
    Hyperexp(WeightedAliasIndex<f64>, Vec<Exp<f64>>, Vec<f64>, Vec<f64>),
}

impl Dist {
    fn new_pareto(alpha: f64) -> Dist {
        assert!(alpha > 1.0);
        // To ensure mean 1.
        let x_m = (alpha - 1.0) / alpha;
        let pareto = Pareto::new(x_m, alpha).unwrap();
        Dist::Pareto(pareto, x_m, alpha)
    }
    fn new_balanced_bimodal(higher: f64) -> Dist {
        assert!(higher > 1.0);
        // higher * p_higher = 1/2
        let p_higher = 1.0 / (2.0 * higher);
        let p_lower = 1.0 - p_higher;
        let lower = 1.0 / (2.0 * p_lower);
        let probabilities = vec![p_lower, p_higher];
        let weighted_index = WeightedAliasIndex::new(probabilities.clone()).unwrap();
        Dist::NModal(weighted_index, probabilities, vec![lower, higher])
    }
    fn new_balanced_hyperexponential(higher: f64) -> Dist {
        assert!(higher > 1.0);
        let p_higher = 1.0 / (2.0 * higher);
        let p_lower = 1.0 - p_higher;
        let lower = 1.0 / (2.0 * p_lower);
        let probs = vec![p_lower, p_higher];
        let mus = vec![1.0 / lower, 1.0 / higher];
        let exps = mus.iter().map(|&mu| Exp::new(mu).unwrap()).collect();
        let weighted_index = WeightedAliasIndex::new(probs.clone()).unwrap();
        Dist::Hyperexp(weighted_index, exps, probs, mus)
    }
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            Dist::Pareto(pareto, _, _) => pareto.sample(rng),
            Dist::NModal(weighted_index, _, choices) => {
                let index = weighted_index.sample(rng);
                choices[index]
            }
            Dist::Hyperexp(weighted_index, exps, _, _) => {
                let index = weighted_index.sample(rng);
                exps[index].sample(rng)
            }
        }
    }
    fn mean(&self) -> f64 {
        match self {
            Dist::Pareto(_, x_m, alpha) => alpha * x_m / (alpha - 1.0),
            Dist::NModal(_, weights, choices) => {
                weights.iter().zip(choices).map(|(w, c)| w * c).sum()
            }
            Dist::Hyperexp(_, _, probs, mus) => probs.iter().zip(mus).map(|(p, mu)| p / mu).sum(),
        }
    }
    fn mean_sq(&self) -> f64 {
        match self {
            Dist::Pareto(_, x_m, alpha) => alpha * x_m.powi(2) / (alpha - 2.0),
            Dist::NModal(_, weights, choices) => weights
                .iter()
                .zip(choices)
                .map(|(w, c)| w * c.powi(2))
                .sum(),
            Dist::Hyperexp(_, _, probs, mus) => probs
                .iter()
                .zip(mus)
                .map(|(p, mu)| 2.0 * p / mu.powi(2))
                .sum(),
        }
    }
    fn mass_above_threshold(&self, threshold: f64) -> f64 {
        match self {
            Dist::Pareto(_, x_m, alpha) => {
                (x_m / threshold).powf(*alpha) * alpha * threshold / (alpha - 1.0)
            }
            Dist::NModal(_, weights, choices) => weights
                .iter()
                .zip(choices)
                .filter(|(_, c)| **c > threshold)
                .map(|(w, c)| w * c)
                .sum(),
            Dist::Hyperexp(_, _, probs, mus) => probs
                .iter()
                .zip(mus)
                .map(|(p, mu)| p * (-mu * threshold).exp() * (mu * threshold + 1.0) / mu)
                .sum(),
        }
    }
}

#[derive(Debug)]
struct Job {
    remaining_size: f64,
    arrival_time: f64,
}

fn simulate(
    dist: Dist,
    num_jobs: u64,
    rho: f64,
    num_servers: usize,
    num_samples: u64,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let lambda = rho;
    let arrival_dist = Exp::new(lambda).unwrap();
    let mut time = 0.0;
    let mut next_arrival_time = time + arrival_dist.sample(&mut rng);
    let mut queue: Vec<Job> = Vec::new();
    let mut samples: Vec<Vec<f64>> = Vec::new();
    let mut num_completed = 0;
    let mut num_arrival = 0;
    let sample_frequency = num_jobs / num_samples;
    while num_arrival < num_jobs {
        let next_event_time = next_arrival_time.min(
            time + queue
                .first()
                .map_or(INFINITY, |job| num_servers as f64 * job.remaining_size),
        );
        let duration = next_event_time - time;
        time = next_event_time;
        for job in queue.iter_mut().take(num_servers) {
            job.remaining_size -= duration / num_servers as f64;
        }
        if time < next_arrival_time {
            let job = queue.remove(0);
            assert!(job.remaining_size < EPSILON);
            num_completed += 1;
        } else {
            num_arrival += 1;
            if num_arrival % sample_frequency == 0 {
                let sizes = queue.iter().map(|job| job.remaining_size).collect();
                samples.push(sizes);
            }
            next_arrival_time = time + arrival_dist.sample(&mut rng);
            let new_size = dist.sample(&mut rng);
            let new_job = Job {
                remaining_size: new_size,
                arrival_time: time,
            };
            let maybe_index =
                queue.binary_search_by_key(&n64(new_size), |job| n64(job.remaining_size));
            let index = match maybe_index {
                Err(index) => index,
                Ok(index) => index,
            };
            queue.insert(index, new_job);
        }
    }
    samples
}

fn main() {
    let rho = 0.999;
    let num_servers = 10;
    let seed = 0;
    let dist = Dist::new_pareto(3.0);
    let num_jobs = 100_000_000;
    let num_samples = 1000;
    println!("rho {} k {} num_jobs {} num_samples {} seed {}",
        rho, num_servers, num_jobs, num_samples, seed);
    println!("{:?}", dist);
    let sample_data = simulate(dist, num_jobs, rho, num_servers, num_samples, seed);
    let mut all_jobs: Vec<f64> = sample_data
        .iter()
        .flat_map(|sample| sample.clone().into_iter())
        .collect();
    all_jobs.sort_by_key(|&size| n64(size));
    let mut total_work = 0.0;
    let fidelity = 1.02;
    let mut some_jobs = vec![];
    let mut some_work = vec![];
    for &size in &all_jobs {
        total_work += size;
        let will_append = some_jobs.last().map_or(true, |old_size| size / old_size > fidelity);
        if will_append {
            some_jobs.push(size);
            some_work.push(total_work / num_samples as f64);
        }
    }
    println!("{}", some_jobs.iter().map(|s| format!("{}", s)).collect::<Vec<String>>().join(","));
    println!("{}", some_work.iter().map(|s| format!("{}", s)).collect::<Vec<String>>().join(","));
}
