use std::{cell::RefCell, collections::HashMap, rc::Rc};

use ndarray::{Array, Array1, Array2, AsArray, Axis, Dimension, Zip};
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};
use ndarray_stats::QuantileExt;

use crate::{
    ch03::{sigmoid::sigmoid, softmax_function::softmax},
    ch04::{cross_entropy_error::cross_entropy_error, gradient::numerical_gradient},
};

#[derive(Clone)]
pub enum Weight {
    M1(Array1<f64>),
    M2(Array2<f64>),
}

impl Weight {
    pub fn unwrap_m1(&self) -> Array1<f64> {
        match self {
            Self::M1(x) => x.clone(),
            Self::M2(_) => panic!(),
        }
    }
    pub fn unwrap_m2(&self) -> Array2<f64> {
        match self {
            Self::M1(_) => panic!(),
            Self::M2(x) => x.clone(),
        }
    }
}

#[derive(Clone)]
pub struct TwoLayerNet {
    params: HashMap<String, Weight>,
}

impl TwoLayerNet {
    fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weight_init_std: f64,
    ) -> TwoLayerNet {
        let w1 = weight_init_std * Array::random((input_size, hidden_size), StandardNormal);
        let b1 = Array1::zeros(hidden_size);
        let w2 = weight_init_std * Array::random((hidden_size, output_size), StandardNormal);
        let b2 = Array1::zeros(output_size);

        let mut params = HashMap::new();
        params.insert("w1".to_owned(), Weight::M2(w1));
        params.insert("b1".to_owned(), Weight::M1(b1));
        params.insert("w2".to_owned(), Weight::M2(w2));
        params.insert("b2".to_owned(), Weight::M1(b2));

        TwoLayerNet { params }
    }
    pub fn predict(&self, x: &Array2<f64>) -> Array2<f64> {
        let w1 = self.params["w1"].unwrap_m2();
        let w2 = self.params["w2"].unwrap_m2();
        let b1 = self.params["b1"].unwrap_m1();
        let b2 = self.params["b2"].unwrap_m1();

        let a1 = x.dot(&w1) + b1;
        let z1 = sigmoid(&a1);
        let a2 = z1.dot(&w2) + b2;
        let y = softmax(&a2);

        y
    }

    pub fn loss(&self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
        let y = self.predict(x);

        cross_entropy_error(&y, t)
    }

    pub fn accuracy(&self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
        let y = self.predict(&x);
        let y = y.map_axis(Axis(0), |y| y.argmax().unwrap());
        let t = t.map_axis(Axis(0), |t| t.argmax().unwrap());
        let accuracy = (y - t).map(|&e| if e == 0 { 1. } else { 0. }).sum() / x.shape()[0] as f64;
        accuracy
    }

    pub fn numerical_gradient(
        &mut self,
        x: &Array2<f64>,
        t: &Array2<f64>,
    ) -> HashMap<&'static str, ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>>
    {
        let net = Rc::new(RefCell::new(self.clone()));
        let list = vec!["w1", "w2", "b1", "b2"];
        let mut grads = HashMap::new();
        for param in list {
            let loss_w = |w: &Array2<f64>| {
                let temp = self.clone();
                self.w1 = w.clone();
                let loss = self.loss(x, t);
                loss
            };
            
            grads.insert(param, numerical_gradient(loss_w, ));
        }

        grads.insert("w1", numerical_gradient(loss_w, &self.w1));
        // grads.insert("b1", numerical_gradient(loss_w, &self.b1.insert_axis(Axis(0))));
        // grads.insert("w2", numerical_gradient(loss_w, &self.w2));
        // grads.insert("b2", numerical_gradient(loss_w, &self.b2.insert_axis(Axis(0))));

        grads
    }
}
