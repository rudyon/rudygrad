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

pub fn make_blobs(n_samples: usize, centers: usize, n_features: usize, cluster_std: f32) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rng = rand::rng();
    let mut inputs = Vec::with_capacity(n_samples);
    let mut targets = Vec::with_capacity(n_samples);

    let mut center_coords = Vec::new();
    for _ in 0..centers {
        let mut center = Vec::with_capacity(n_features);
        for _ in 0..n_features {
            // Keep centers within the bounds that demo.rs renders (-1.5 to 2.5) 
            // We use -1.0 to 1.0 to be safe
            center.push(rng.random_range(-1.0..1.0));
        }
        center_coords.push(center);
    }

    let samples_per_center = n_samples / centers;
    let mut remaining = n_samples;

    for (i, center) in center_coords.iter().enumerate() {
        let n = if i == centers - 1 { remaining } else { samples_per_center };
        remaining -= n;
        
        // For binary classification compatibility with margin loss
        let label = if centers == 2 {
            if i == 0 { -1.0 } else { 1.0 }
        } else {
            i as f32
        };

        for _ in 0..n {
            let mut point = Vec::with_capacity(n_features);
            for d in 0..n_features {
                point.push(center[d] + rand_normal(&mut rng) * cluster_std);
            }
            inputs.push(point);
            targets.push(label);
        }
    }
    
    (inputs, targets)
}