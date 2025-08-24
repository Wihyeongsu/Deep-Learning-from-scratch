use ndarray::*;
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};
use ndarray_stats::QuantileExt;

use crate::ch03::mnist_dataset::{load_mnist, MnistDataset};

use super::{sigmoid::sigmoid, softmax_function::softmax};

pub struct MnistNetworkBatch {
    w: Vec<Array2<f64>>,
    b: Vec<Array1<f64>>,
}

impl MnistNetworkBatch {
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
        MnistNetworkBatch { w, b }
    }
    fn predict(&self, x: &Array2<f64>) -> Array2<f64> {
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
    } = load_mnist((60_000, 0, 10_000), true, false);
    let network = MnistNetworkBatch::new();
    let batch_size = 100;
    let mut accuracy_cnt = 0_usize;
    for i in 0..x_train_2d.nrows() / batch_size {
        let i_batch = i * batch_size;
        let x_batch = x_train_2d.slice(s![i_batch..i_batch + batch_size, ..]).into_owned();
        let y_batch = network.predict(&x_batch);
        let p = y_batch
            .rows()
            .into_iter()
            .map(|y| y.argmax().unwrap() as f64)
            .collect::<Array1<f64>>();
        accuracy_cnt += (&p
            - &t_train.slice(s![i_batch..i_batch + batch_size, ..])
                .into_owned()
                .into_flat())
            .mapv(|t| if t == 0. { 1 } else { 0 })
            .sum();
        println!(
            "[{}/{}] Accuracy: {:.2}%",
            i + 1,
            x_train_2d.nrows() / batch_size,
            accuracy_cnt as f64 / (i_batch + batch_size) as f64 * 100.
        );
    }
    println!(
        "Accuracy: {:.2}%",
        accuracy_cnt as f64 / x_train_2d.nrows() as f64 * 100.
    );
}
