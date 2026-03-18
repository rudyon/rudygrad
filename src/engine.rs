use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;

pub struct ValueData {
    pub data: f32,
    pub grad: f32,
    pub _prev: Vec<Value>,
    pub _backward: Option<Box<dyn Fn(f32)>>,
}

#[derive(Clone)]
pub struct Value(pub Rc<RefCell<ValueData>>);

impl Value {
    pub fn new(data: f32) -> Self {
        Value(Rc::new(RefCell::new(ValueData {
            data,
            grad: 0.0,
            _prev: Vec::new(),
            _backward: None,
        })))
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(
            v: &Value,
            visited: &mut HashSet<*const RefCell<ValueData>>,
            topo: &mut Vec<Value>,
        ) {
            let ptr = Rc::as_ptr(&v.0);
            if !visited.contains(&ptr) {
                visited.insert(ptr);
                for child in &v.0.borrow()._prev {
                    build_topo(child, visited, topo);
                }
                topo.push(v.clone());
            }
        }

        build_topo(self, &mut visited, &mut topo);
        self.0.borrow_mut().grad = 1.0;

        for node in topo.iter().rev() {
            let out_grad = node.0.borrow().grad;
            if let Some(ref backward_fn) = node.0.borrow()._backward {
                backward_fn(out_grad);
            }
        }
    }

    pub fn negate(&self) -> Value {
        self.clone() * Value::new(-1.0)
    }

    pub fn pow(&self, other: f32) -> Value {
        let data = self.0.borrow().data.powf(other);
        let out = Value::new(data);
        
        out.0.borrow_mut()._prev = vec![self.clone()];

        let self_clone = self.clone();
        
        
        out.0.borrow_mut()._backward = Some(Box::new(move |out_grad| {
            let self_data = self_clone.0.borrow().data;
            self_clone.0.borrow_mut().grad += (other * self_data.powf(other - 1.0)) * out_grad;
        }));
        out
    }

    pub fn relu(&self) -> Value {
        let x = self.0.borrow().data;
        let out_data = if x > 0.0 { x } else { 0.0 };
        let out = Value::new(out_data);
        
        out.0.borrow_mut()._prev = vec![self.clone()];

        let self_clone = self.clone();
        
        out.0.borrow_mut()._backward = Some(Box::new(move |out_grad| {
            self_clone.0.borrow_mut().grad += (if out_data > 0.0 { 1.0 } else { 0.0 }) * out_grad;
        }));
        out
    }

    pub fn tanh(&self) -> Value {
        let x = self.0.borrow().data;
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
        let out = Value::new(t);

        out.0.borrow_mut()._prev = vec![self.clone()];

        let self_clone = self.clone();

        out.0.borrow_mut()._backward = Some(Box::new(move |out_grad| {
            self_clone.0.borrow_mut().grad += (1.0 - t.powi(2)) * out_grad;
        }));
        out
    }

    pub fn dot(w: &[Value], x: &[Value], b: &Value) -> Value {
        let mut data = b.0.borrow().data;
        for (wi, xi) in w.iter().zip(x.iter()) {
            data += wi.0.borrow().data * xi.0.borrow().data;
        }
        let out = Value::new(data);
        
        let mut prev = w.to_vec();
        prev.extend_from_slice(x);
        prev.push(b.clone());
        out.0.borrow_mut()._prev = prev;

        let w_clone = w.to_vec();
        let x_clone = x.to_vec();
        let b_clone = b.clone();

        out.0.borrow_mut()._backward = Some(Box::new(move |out_grad| {
            for (wi, xi) in w_clone.iter().zip(x_clone.iter()) {
                let w_data = wi.0.borrow().data;
                let x_data = xi.0.borrow().data;
                wi.0.borrow_mut().grad += x_data * out_grad;
                xi.0.borrow_mut().grad += w_data * out_grad;
            }
            b_clone.0.borrow_mut().grad += out_grad;
        }));
        out
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        let out = Value::new(self.0.borrow().data + other.0.borrow().data);
        
        out.0.borrow_mut()._prev = vec![self.clone(), other.clone()];

        let self_clone = self.clone();
        let other_clone = other.clone();
        
        out.0.borrow_mut()._backward = Some(Box::new(move |out_grad| {
            self_clone.0.borrow_mut().grad += out_grad;
            other_clone.0.borrow_mut().grad += out_grad;
        }));
        out
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        let out = Value::new(self.0.borrow().data * other.0.borrow().data);
        
        out.0.borrow_mut()._prev = vec![self.clone(), other.clone()];

        let self_clone = self.clone();
        let other_clone = other.clone();

        out.0.borrow_mut()._backward = Some(Box::new(move |out_grad| {
            let self_data = self_clone.0.borrow().data;
            let other_data = other_clone.0.borrow().data;
            
            self_clone.0.borrow_mut().grad += other_data * out_grad;
            other_clone.0.borrow_mut().grad += self_data * out_grad;
        }));
        out
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        self + other.neg()
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        self.negate()
    }
}