use std::hash::BuildHasherDefault;

use ndarray::*;
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};
use ndarray_stats::QuantileExt;

use crate::{
    ch03::mnist_dataset::{MnistDataset, load_mnist},
    common::bigfloat_array::BigFloatArray,
};

use super::{sigmoid::sigmoid, softmax_function::*};

pub struct MnistNetwork {
    w: Vec<BigFloatArray<Ix2>>,
    b: Vec<BigFloatArray<Ix1>>,
}

impl MnistNetwork {
    pub fn new() -> Self {
        let w = vec![
            BigFloatArray::from(Array::random((784, 50), StandardNormal).mapv(|x: f64| x)),
            BigFloatArray::from(Array::random((50, 100), StandardNormal).mapv(|x: f64| x)),
            BigFloatArray::from(Array::random((100, 10), StandardNormal).mapv(|x: f64| x)),
        ];
        let b = vec![
            BigFloatArray::from(Array::random(50, StandardNormal).mapv(|x: f64| x)),
            BigFloatArray::from(Array::random(100, StandardNormal).mapv(|x: f64| x)),
            BigFloatArray::from(Array::random(10, StandardNormal).mapv(|x: f64| x)),
        ];
        MnistNetwork { w, b }
    }
    fn predict(&self, x: &Array1<Element>) -> Array1<Element> {
        let a1 = x.dot(&self.w[0]) + &self.b[0];
        let z1 = sigmoid(&a1);
        let a2 = z1.dot(&self.w[1]) + &self.b[1];
        let z2 = sigmoid(&a2);
        let a3 = z2.dot(&self.w[2]) + &self.b[2];
        let y = softmax(&a3);

        y
    }
}

pub fn run() {
    let MnistDataset {
        x_train_2d,
        t_train,
        ..
    } = load_mnist((60_000, 0, 10_000), true, true);
    let network = MnistNetwork::new();
    let mut accuracy_cnt = 0;
    for i in 0..x_train_2d.nrows() {
        let y = network.predict(&x_train_2d.row(i).into_owned());
        let p = y.argmax().unwrap();
        if p == t_train[[i, 0]] {
            accuracy_cnt += 1;
        }
        println!(
            "[{}/{}] (y, target): ({}, {}) Accuracy: {:.2}%",
            i + 1,
            x_train_2d.nrows(),
            p,
            t_train[[i, 0]],
            accuracy_cnt as Element / (i + 1) as Element * 100.
        );
    }
    println!(
        "Accuracy: {:.2}%",
        accuracy_cnt as Element / x_train_2d.nrows() as Element * 100.
    );
}
