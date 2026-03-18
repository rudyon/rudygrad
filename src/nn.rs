use crate::engine::Value;
use rand::distr::{Distribution, Uniform};

pub struct Neuron {
    pub w: Vec<Value>,
    pub b: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::rng();
        let range = Uniform::new(-1.0f32, 1.0f32).unwrap();
        let w = (0..nin)
            .map(|_| Value::new(range.sample(&mut rng)))
            .collect();
        let b = Value::new(range.sample(&mut rng));
        Neuron { w, b }
    }

    pub fn call(&self, x: &[Value]) -> Value {
        let mut act = self.b.clone();
        for (wi, xi) in self.w.iter().zip(x.iter()) {
            act = act + (wi.clone() * xi.clone());
        }
        act.tanh()
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.w.clone();
        p.push(self.b.clone());
        p
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin)).collect();
        Layer { neurons }
    }
    pub fn call(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.call(x)).collect()
    }
    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: Vec<usize>) -> Self {
        let mut sz = vec![nin];
        sz.extend(nouts);
        let layers = (0..sz.len() - 1)
            .map(|i| Layer::new(sz[i], sz[i+1]))
            .collect();
        MLP { layers }
    }
    pub fn call(&self, mut x: Vec<Value>) -> Vec<Value> {
        for layer in &self.layers {
            x = layer.call(&x);
        }
        x
    }
    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.0.borrow_mut().grad = 0.0;
        }
    }
}