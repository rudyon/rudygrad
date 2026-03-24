use pyo3::prelude::*;
use crate::engine::Value;
use crate::nn::MLP;

#[pyclass(name = "Value", unsendable)]
#[derive(Clone)]
pub struct PyValue {
    pub inner: Value,
}

#[pymethods]
impl PyValue {
    #[new]
    fn new(data: f32) -> Self {
        PyValue { inner: Value::new(data) }
    }

    #[getter]
    fn data(&self) -> f32 {
        self.inner.0.borrow().data
    }

    #[setter]
    fn set_data(&self, data: f32) {
        self.inner.0.borrow_mut().data = data;
    }

    #[getter]
    fn grad(&self) -> f32 {
        self.inner.0.borrow().grad
    }

    #[setter]
    fn set_grad(&self, grad: f32) {
        self.inner.0.borrow_mut().grad = grad;
    }

    fn backward(&self) {
        self.inner.backward();
    }

    fn relu(&self) -> Self {
        PyValue { inner: self.inner.relu() }
    }

    fn tanh(&self) -> Self {
        PyValue { inner: self.inner.tanh() }
    }

    fn __add__(&self, other: &PyValue) -> Self {
        PyValue { inner: self.inner.clone() + other.inner.clone() }
    }

    fn __mul__(&self, other: &PyValue) -> Self {
        PyValue { inner: self.inner.clone() * other.inner.clone() }
    }

    fn __sub__(&self, other: &PyValue) -> Self {
        PyValue { inner: self.inner.clone() - other.inner.clone() }
    }

    fn __str__(&self) -> String {
        format!("Value(data={}, grad={})", self.data(), self.grad())
    }
    
    fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pyclass(name = "MLP", unsendable)]
pub struct PyMLP {
    pub inner: MLP,
}

#[pymethods]
impl PyMLP {
    #[new]
    fn new(nin: usize, nouts: Vec<usize>) -> Self {
        PyMLP { inner: MLP::new(nin, nouts) }
    }

    fn call(&self, x: Vec<PyValue>) -> Vec<PyValue> {
        let inner_x: Vec<Value> = x.into_iter().map(|v| v.inner).collect();
        let out = self.inner.call(inner_x);
        out.into_iter().map(|v| PyValue { inner: v }).collect()
    }

    fn parameters(&self) -> Vec<PyValue> {
        self.inner.parameters().iter().map(|v| PyValue { inner: v.clone() }).collect()
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }
}

#[pymodule]
fn rudygrad(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyValue>()?;
    m.add_class::<PyMLP>()?;
    Ok(())
}
