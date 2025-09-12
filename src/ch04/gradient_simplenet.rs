use std::{cell::RefCell, rc::Rc};

use ndarray::{Array, Array1, Array2, Dimension};
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};

use crate::{ch03::softmax_function::softmax, ch04::cross_entropy_error::cross_entropy_error};

#[derive(Clone)]
pub struct SimpleNet {
    pub w: Array2<f64>, // RefCell
}

impl SimpleNet {
    pub fn new() -> Self {
        Self {
            w: Array::random((2, 3), StandardNormal),
        }
    }

    pub fn predict(&self, x: &Array1<f64>) -> Array1<f64> {
        x.dot(&self.w)
    }

    pub fn loss(&self, x: &Array1<f64>, t: &Array1<f64>) -> f64 {
        let z = self.predict(x);
        let y = softmax(&z);
        let loss = cross_entropy_error(&y, t);

        loss
    }

    pub fn loss_with_weights(&mut self, w: &Array2<f64>, x: &Array1<f64>, t: &Array1<f64>) -> f64 {
        let temp_w = self.w.clone();
        self.w = w.clone();
        let loss = self.loss(x, t);
        self.w = temp_w;
        loss
    }
}
