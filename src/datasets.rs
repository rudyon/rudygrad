use rand::RngExt;
use std::f32::consts::PI;
use crate::util::rand_normal;

pub fn make_moons(n_samples: usize, noise: f32) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rng = rand::rng();
    let mut inputs = Vec::with_capacity(n_samples);
    let mut targets = Vec::with_capacity(n_samples);

    let n_out = n_samples / 2;
    let n_in = n_samples - n_out;

    for _ in 0..n_out {
        let theta = rng.random_range(0.0..PI);
        let nx = rand_normal(&mut rng) * noise;
        let ny = rand_normal(&mut rng) * noise;
        inputs.push(vec![theta.cos() + nx, theta.sin() + ny]);
        targets.push(-1.0); 
    }

    for _ in 0..n_in {
        let theta = rng.random_range(0.0..PI);
        let nx = rand_normal(&mut rng) * noise;
        let ny = rand_normal(&mut rng) * noise;
        inputs.push(vec![1.0 - theta.cos() + nx, 1.0 - theta.sin() - 0.5 + ny]);
        targets.push(1.0);
    }
    (inputs, targets)
}