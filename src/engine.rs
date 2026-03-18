use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;

pub enum Op {
    Add,
    Mul,
    Pow(f32),
    ReLU,
    Tanh,
    Dot(usize),
}

pub struct ValueData {
    pub data: f32,
    pub grad: f32,
    pub _prev: Vec<Value>,
    pub _op: Option<Op>,
}

#[derive(Clone)]
pub struct Value(pub Rc<RefCell<ValueData>>);

impl Value {
    pub fn new(data: f32) -> Self {
        Value(Rc::new(RefCell::new(ValueData {
            data,
            grad: 0.0,
            _prev: Vec::new(),
            _op: None,
        })))
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![(self.clone(), false)];

        while let Some((v, processed)) = stack.pop() {
            let ptr = Rc::as_ptr(&v.0);
            if processed {
                topo.push(v);
            } else if !visited.contains(&ptr) {
                visited.insert(ptr);
                stack.push((v.clone(), true));
                for child in &v.0.borrow()._prev {
                    stack.push((child.clone(), false));
                }
            }
        }

        self.0.borrow_mut().grad = 1.0;

        for node in topo.iter().rev() {
            let out_grad = node.0.borrow().grad;
            let node_borrow = node.0.borrow();
            if let Some(ref op) = node_borrow._op {
                match op {
                    Op::Add => {
                        node_borrow._prev[0].0.borrow_mut().grad += out_grad;
                        node_borrow._prev[1].0.borrow_mut().grad += out_grad;
                    }
                    Op::Mul => {
                        let d0 = node_borrow._prev[0].0.borrow().data;
                        let d1 = node_borrow._prev[1].0.borrow().data;
                        node_borrow._prev[0].0.borrow_mut().grad += d1 * out_grad;
                        node_borrow._prev[1].0.borrow_mut().grad += d0 * out_grad;
                    }
                    Op::Pow(other) => {
                        let d0 = node_borrow._prev[0].0.borrow().data;
                        node_borrow._prev[0].0.borrow_mut().grad += (other * d0.powf(other - 1.0)) * out_grad;
                    }
                    Op::ReLU => {
                        let d = node_borrow.data;
                        node_borrow._prev[0].0.borrow_mut().grad += (if d > 0.0 { 1.0 } else { 0.0 }) * out_grad;
                    }
                    Op::Tanh => {
                        let d = node_borrow.data;
                        node_borrow._prev[0].0.borrow_mut().grad += (1.0 - d * d) * out_grad;
                    }
                    Op::Dot(nin) => {
                        let nin = *nin;
                        for i in 0..nin {
                            let w_data = node_borrow._prev[i].0.borrow().data;
                            let x_data = node_borrow._prev[nin + i].0.borrow().data;
                            node_borrow._prev[i].0.borrow_mut().grad += x_data * out_grad;
                            node_borrow._prev[nin + i].0.borrow_mut().grad += w_data * out_grad;
                        }
                        node_borrow._prev[2 * nin].0.borrow_mut().grad += out_grad;
                    }
                }
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
        out.0.borrow_mut()._op = Some(Op::Pow(other));
        out
    }

    pub fn relu(&self) -> Value {
        let x = self.0.borrow().data;
        let out_data = if x > 0.0 { x } else { 0.0 };
        let out = Value::new(out_data);
        out.0.borrow_mut()._prev = vec![self.clone()];
        out.0.borrow_mut()._op = Some(Op::ReLU);
        out
    }

    pub fn tanh(&self) -> Value {
        let x = self.0.borrow().data;
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
        let out = Value::new(t);
        out.0.borrow_mut()._prev = vec![self.clone()];
        out.0.borrow_mut()._op = Some(Op::Tanh);
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
        out.0.borrow_mut()._op = Some(Op::Dot(w.len()));
        out
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        let out = Value::new(self.0.borrow().data + other.0.borrow().data);
        out.0.borrow_mut()._prev = vec![self.clone(), other.clone()];
        out.0.borrow_mut()._op = Some(Op::Add);
        out
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        let out = Value::new(self.0.borrow().data * other.0.borrow().data);
        out.0.borrow_mut()._prev = vec![self.clone(), other.clone()];
        out.0.borrow_mut()._op = Some(Op::Mul);
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