use std::{
    cell::RefCell, collections::HashMap, rc::Rc,
};

use ndarray::{Array, Array1, Array2, Axis, Ix1, Ix2};
use ndarray_rand::{
    RandomExt,
    rand::{self, seq::index::sample},
    rand_distr::StandardNormal,
};
use ndarray_stats::QuantileExt;

use crate::{
    ch03::{
        mnist_dataset::{MnistDataset, load_mnist},
        sigmoid::sigmoid,
        softmax_function::softmax,
    },
    ch04::{cross_entropy_error::cross_entropy_error, gradient::numerical_gradient},
};

#[derive(Clone, Debug)]
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
    loss: Option<f64>, // cache loss to avoid recomputation
}

impl TwoLayerNet {
    pub fn new(
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

        TwoLayerNet { params, loss: None }
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

    pub fn loss(&mut self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
        if self.loss.is_none() {
            let y = self.predict(x);
            self.loss = Some(cross_entropy_error(&y, t));
        }
        self.loss.unwrap()
    }

    pub fn reset_loss(&mut self) {
        self.loss = None;
    }

    pub fn accuracy(&self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
        let y = self.predict(&x);
        let y = y.map_axis(Axis(0), |y| y.argmax().unwrap());
        let t = t.map_axis(Axis(0), |t| t.argmax().unwrap());
        let accuracy = (y - t).map(|&e| if e == 0 { 1. } else { 0. }).sum() / x.shape()[0] as f64;
        accuracy
    }

    pub fn numerical_gradient(&self, x: &Array2<f64>, t: &Array2<f64>) -> HashMap<String, Weight> {
        let net = Rc::new(RefCell::new(self.clone()));
        let mut grads = HashMap::new();
        for num_param in 1..=self.params.len() / 2 {
            let w_key = format!("w{}", num_param);
            let w = &self.params.get(&w_key).unwrap().unwrap_m2();
            let b_key = format!("b{}", num_param);
            let b = &self.params.get(&b_key).unwrap().unwrap_m1();

            let loss_w = |w: &Array2<f64>| {
                let net = net.clone();
                let temp = net.borrow().params.get(&w_key).unwrap().unwrap_m2();
                net.borrow_mut()
                    .params
                    .insert(w_key.clone(), Weight::M2(w.clone()));
                let loss = net.borrow_mut().loss(x, t);
                net.borrow_mut()
                    .params
                    .insert(w_key.clone(), Weight::M2(temp));
                loss
            };
            let loss_b = |b: &Array1<f64>| {
                let net = net.clone();
                let temp = net.borrow().params.get(&b_key).unwrap().unwrap_m1();
                net.borrow_mut()
                    .params
                    .insert(b_key.clone(), Weight::M1(b.clone()));
                let loss = net.borrow_mut().loss(x, t);
                net.borrow_mut()
                    .params
                    .insert(b_key.clone(), Weight::M1(temp));
                loss
            };

            grads.insert(
                w_key.clone(),
                Weight::M2(numerical_gradient::<_, Ix2>(loss_w, w)),
            );
            grads.insert(
                b_key.clone(),
                Weight::M1(numerical_gradient::<_, Ix1>(loss_b, b)),
            );
        }

        grads
    }
}

pub fn mini_batch() {
    let MnistDataset {
        x_train_2d,
        t_train,
        x_test_2d,
        t_test,
        ..
    } = load_mnist((60_000, 0, 10_000), true, true);

    let mut train_loss_list = Vec::new();

    let iters_num = 10000;
    let train_size = x_train_2d.shape()[0];
    let batch_size = 100;
    let learning_rate = 0.1;

    let mut network = TwoLayerNet::new(784, 50, 10, 0.01);

    for i in 1..=iters_num {
        // mini batch
        let batch_mask = sample(&mut rand::thread_rng(), train_size, batch_size).into_vec();
        let x_batch = x_train_2d.select(Axis(0), &batch_mask);
        let t_batch = t_train.select(Axis(0), &batch_mask);

        // gradient
        network.reset_loss();
        let mut grad = network.numerical_gradient(&x_batch, &t_batch);

        // update parameters
        for key in ["w1", "b1", "w2", "b2"] {
            match grad.get(key) {
                Some(Weight::M1(b)) => {
                    let mut param = network.params.get_mut(key).unwrap().unwrap_m1();
                    param = param - learning_rate * b;
                }
                Some(Weight::M2(w)) => {
                    let mut param = network.params.get_mut(key).unwrap().unwrap_m2();
                    param = param - learning_rate * w;
                }
                _ => {
                    panic!()
                }
            }
        }

        // loss
        let loss = network.loss(&x_batch, &t_batch);
        train_loss_list.push(loss);

        println!(
            "Batch ({i}/{iters_num}): Loss = {loss} Accuracy = {accuracy}",
            accuracy = network.accuracy(&x_batch, &t_batch)
        );
    }

    // print loss
    dbg!(train_loss_list);
}
