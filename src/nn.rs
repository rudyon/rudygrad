use crate::engine::Value;
use rand::distr::{Distribution, Uniform};
use std::rc::Rc;

pub struct Neuron {
    pub w: Vec<Value>,
    pub b: Value,
    pub params: Vec<Value>,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::rng();
        let range = Uniform::new(-1.0f32, 1.0f32).unwrap();
        let w: Vec<Value> = (0..nin)
            .map(|_| Value::new(range.sample(&mut rng)))
            .collect();
        let b = Value::new(range.sample(&mut rng));
        let mut params = w.clone();
        params.push(b.clone());
        Neuron { w, b, params }
    }

    pub fn call(&self, x: &Rc<Vec<Value>>) -> Value {
        Value::dot_tanh(&self.w, x, &self.b)
    }

    pub fn parameters(&self) -> &[Value] {
        &self.params
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub params: Vec<Value>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let neurons: Vec<Neuron> = (0..nout).map(|_| Neuron::new(nin)).collect();
        let params = neurons.iter().flat_map(|n| n.parameters().to_vec()).collect();
        Layer { neurons, params }
    }
    pub fn call(&self, x: &Rc<Vec<Value>>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.call(x)).collect()
    }
    pub fn parameters(&self) -> &[Value] {
        &self.params
    }
}

pub struct MLP {
    pub layers: Vec<Layer>,
    pub params: Vec<Value>,
}

impl MLP {
    pub fn new(nin: usize, nouts: Vec<usize>) -> Self {
        let mut sz = vec![nin];
        sz.extend(nouts);
        let layers: Vec<Layer> = (0..sz.len() - 1)
            .map(|i| Layer::new(sz[i], sz[i+1]))
            .collect();
        let params = layers.iter().flat_map(|l| l.parameters().to_vec()).collect();
        MLP { layers, params }
    }
    pub fn call(&self, mut x: Vec<Value>) -> Vec<Value> {
        for layer in &self.layers {
            let x_rc = Rc::new(x);
            x = layer.call(&x_rc);
        }
        x
    }
    pub fn parameters(&self) -> &[Value] {
        &self.params
    }
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.0.borrow_mut().grad = 0.0;
        }
    }
}
