use std::cell::UnsafeCell;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static BACKWARD_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy)]
pub enum Op {
    Add,
    Mul,
    Pow(f32),
    ReLU,
    Tanh,
    Dot(u32),
    DotTanh(u32),
}

#[derive(Clone)]
pub enum Prev {
    None,
    One(Value),
    Two(Value, Value),
    Many(Rc<Vec<Value>>),
    Dot {
        w: Rc<Vec<Value>>,
        x: Rc<Vec<Value>>,
        b: Value,
    },
}

pub struct FastCell<T>(pub Rc<UnsafeCell<T>>);

impl<T> FastCell<T> {
    pub fn borrow(&self) -> &T {
        unsafe { &*self.0.get() }
    }
    pub fn borrow_mut(&self) -> &mut T {
        unsafe { &mut *self.0.get() }
    }
}

impl<T> Clone for FastCell<T> {
    fn clone(&self) -> Self {
        FastCell(self.0.clone())
    }
}

pub struct ValueData {
    pub data: f32,
    pub grad: f32,
    pub _prev: Prev,
    pub _op: Option<Op>,
    pub visited_at: u32,
}

#[derive(Clone)]
pub struct Value(pub FastCell<ValueData>);

impl Value {
    pub fn new(data: f32) -> Self {
        Value(FastCell(Rc::new(UnsafeCell::new(ValueData {
            data,
            grad: 0.0,
            _prev: Prev::None,
            _op: None,
            visited_at: 0,
        }))))
    }

    pub fn dot_tanh(w: &Rc<Vec<Value>>, x: &Rc<Vec<Value>>, b: &Value) -> Value {
        let mut data = b.0.borrow().data;
        for (wi, xi) in w.iter().zip(x.iter()) {
            data += wi.0.borrow().data * xi.0.borrow().data;
        }
        let e2x = (2.0 * data).exp();
        let t = (e2x - 1.0) / (e2x + 1.0);
        let out = Value::new(t);
        out.0.borrow_mut()._prev = Prev::Dot { w: w.clone(), x: x.clone(), b: b.clone() };
        out.0.borrow_mut()._op = Some(Op::DotTanh(w.len() as u32));
        out
    }

    pub fn dot(w: &Rc<Vec<Value>>, x: &Rc<Vec<Value>>, b: &Value) -> Value {
        let mut data = b.0.borrow().data;
        for (wi, xi) in w.iter().zip(x.iter()) {
            data += wi.0.borrow().data * xi.0.borrow().data;
        }
        let out = Value::new(data);
        out.0.borrow_mut()._prev = Prev::Dot { w: w.clone(), x: x.clone(), b: b.clone() };
        out.0.borrow_mut()._op = Some(Op::Dot(w.len() as u32));
        out
    }

    pub fn backward(&self) {
        let mut topo: Vec<*mut ValueData> = Vec::with_capacity(4096);
        let counter = (BACKWARD_COUNTER.fetch_add(1, Ordering::Relaxed) + 1) as u32;
        let mut stack: Vec<(*mut ValueData, bool)> = Vec::with_capacity(4096);
        stack.push((self.0.0.get(), false));

        while let Some((v_ptr, processed)) = stack.pop() {
            if processed {
                topo.push(v_ptr);
            } else {
                unsafe {
                    if (*v_ptr).visited_at != counter {
                        (*v_ptr).visited_at = counter;
                        stack.push((v_ptr, true));
                        match &(*v_ptr)._prev {
                            Prev::None => (),
                            Prev::One(a) => stack.push((a.0.0.get(), false)),
                            Prev::Two(a, b) => {
                                stack.push((a.0.0.get(), false));
                                stack.push((b.0.0.get(), false));
                            }
                            Prev::Many(vec) => {
                                for child in vec.iter() {
                                    stack.push((child.0.0.get(), false));
                                }
                            }
                            Prev::Dot { w, x, b } => {
                                for child in w.iter() { stack.push((child.0.0.get(), false)); }
                                for child in x.iter() { stack.push((child.0.0.get(), false)); }
                                stack.push((b.0.0.get(), false));
                            }
                        }
                    }
                }
            }
        }

        unsafe { (*self.0.0.get()).grad = 1.0; }

        for &node_ptr in topo.iter().rev() {
            unsafe {
                let out_grad = (*node_ptr).grad;
                if out_grad == 0.0 { continue; }
                
                if let Some(op) = (*node_ptr)._op {
                    match op {
                        Op::Add => {
                            if let Prev::Two(ref a, ref b) = (*node_ptr)._prev {
                                (*a.0.0.get()).grad += out_grad;
                                (*b.0.0.get()).grad += out_grad;
                            }
                        }
                        Op::Mul => {
                            if let Prev::Two(ref a, ref b) = (*node_ptr)._prev {
                                let d0 = (*a.0.0.get()).data;
                                let d1 = (*b.0.0.get()).data;
                                (*a.0.0.get()).grad += d1 * out_grad;
                                (*b.0.0.get()).grad += d0 * out_grad;
                            }
                        }
                        Op::Pow(other) => {
                            if let Prev::One(ref a) = (*node_ptr)._prev {
                                let d0 = (*a.0.0.get()).data;
                                (*a.0.0.get()).grad += (other * d0.powf(other - 1.0)) * out_grad;
                            }
                        }
                        Op::ReLU => {
                            if let Prev::One(ref a) = (*node_ptr)._prev {
                                let d = (*node_ptr).data;
                                if d > 0.0 {
                                    (*a.0.0.get()).grad += out_grad;
                                }
                            }
                        }
                        Op::Tanh => {
                            if let Prev::One(ref a) = (*node_ptr)._prev {
                                let d = (*node_ptr).data;
                                let local_grad = 1.0 - d * d;
                                if local_grad != 0.0 {
                                    (*a.0.0.get()).grad += local_grad * out_grad;
                                }
                            }
                        }
                        Op::Dot(nin) => {
                            if let Prev::Dot { ref w, ref x, ref b } = (*node_ptr)._prev {
                                for i in 0..nin as usize {
                                    let w_ptr = w[i].0.0.get();
                                    let x_ptr = x[i].0.0.get();
                                    (*w_ptr).grad += (*x_ptr).data * out_grad;
                                    (*x_ptr).grad += (*w_ptr).data * out_grad;
                                }
                                (*b.0.0.get()).grad += out_grad;
                            }
                        }
                        Op::DotTanh(nin) => {
                            if let Prev::Dot { ref w, ref x, ref b } = (*node_ptr)._prev {
                                let d = (*node_ptr).data;
                                let local_grad = (1.0 - d * d) * out_grad;
                                for i in 0..nin as usize {
                                    let w_ptr = w[i].0.0.get();
                                    let x_ptr = x[i].0.0.get();
                                    (*w_ptr).grad += (*x_ptr).data * local_grad;
                                    (*x_ptr).grad += (*w_ptr).data * local_grad;
                                }
                                (*b.0.0.get()).grad += local_grad;
                            }
                        }
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
        out.0.borrow_mut()._prev = Prev::One(self.clone());
        out.0.borrow_mut()._op = Some(Op::Pow(other));
        out
    }

    pub fn relu(&self) -> Value {
        let x = self.0.borrow().data;
        let out_data = if x > 0.0 { x } else { 0.0 };
        let out = Value::new(out_data);
        out.0.borrow_mut()._prev = Prev::One(self.clone());
        out.0.borrow_mut()._op = Some(Op::ReLU);
        out
    }

    pub fn tanh(&self) -> Value {
        let x: f32 = self.0.borrow().data;
        let e2x = (2.0 * x).exp();
        let t = (e2x - 1.0) / (e2x + 1.0);
        let out = Value::new(t);
        out.0.borrow_mut()._prev = Prev::One(self.clone());
        out.0.borrow_mut()._op = Some(Op::Tanh);
        out
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        let out = Value::new(self.0.borrow().data + other.0.borrow().data);
        out.0.borrow_mut()._prev = Prev::Two(self.clone(), other.clone());
        out.0.borrow_mut()._op = Some(Op::Add);
        out
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        let out = Value::new(self.0.borrow().data * other.0.borrow().data);
        out.0.borrow_mut()._prev = Prev::Two(self.clone(), other.clone());
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
