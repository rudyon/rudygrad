use rand::{Rng, RngExt};
use std::f32::consts::PI;

pub fn rand_normal(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.random_range(0.0001..1.0);
    let u2: f32 = rng.random_range(0.0001..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}