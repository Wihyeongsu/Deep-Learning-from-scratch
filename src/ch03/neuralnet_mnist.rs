use ndarray::*;
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};
use ndarray_stats::QuantileExt;

use crate::ch03::mnist_dataset::{load_mnist, MnistDataset};

use super::{sigmoid::sigmoid, softmax_function::*};

pub struct MnistNetwork {
    w: Vec<Array2<f64>>,
    b: Vec<Array1<f64>>,
}

impl MnistNetwork {
    pub fn new() -> Self {
        let w = vec![
            Array::random((784, 50), StandardNormal),
            Array::random((50, 100), StandardNormal),
            Array::random((100, 10), StandardNormal),
        ];
        let b = vec![
            Array::random(50, StandardNormal),
            Array::random(100, StandardNormal),
            Array::random(10, StandardNormal),
        ];
        MnistNetwork { w, b }
    }
    fn predict(&self, x: &Array1<f64>) -> Array1<f64> {
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
    let MnistDataset{
        x_train_2d,
        t_train,
        ..
    } = load_mnist((60_000, 0, 10_000), true, true);
    let network = MnistNetwork::new();
    let mut accuracy_cnt = 0;
    for i in 0..x_train_2d.nrows() {
        let y = network.predict(&x_train_2d.row(i).into_owned());
        let p = y.argmax().unwrap() as f64;
        if p == t_train[[i, 0]] {
            accuracy_cnt += 1;
        }
        println!(
            "[{}/{}] (y, target): ({}, {}) Accuracy: {:.2}%",
            i + 1,
            x_train_2d.nrows(),
            p,
            t_train[[i, 0]],
            accuracy_cnt as f64 / (i + 1) as f64 * 100.
        );
    }
    println!(
        "Accuracy: {:.2}%",
        accuracy_cnt as f64 / x_train_2d.nrows() as f64 * 100.
    );
}
