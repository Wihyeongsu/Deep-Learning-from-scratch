use mnist::*;
use ndarray::*;
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};
use ndarray_stats::QuantileExt;

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

pub fn get_data() -> (Array2<f64>, Array2<f64>) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("data/")
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let x_train = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);
    println!("{:#.1?}\n", x_train.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let t_train: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    println!(
        "The first digit is a {:?}",
        t_train.slice(s![image_num, ..])
    );

    let x_test = Array2::from_shape_vec((10_000, 784), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.);

    let t_test: Array2<f64> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f64);

    (x_test, t_test)
}

pub fn run() {
    let (x, t) = get_data();
    let network = MnistNetwork::new();
    let mut accuracy_cnt = 0;
    for i in 0..x.nrows() {
        let y = network.predict(&x.row(i).into_owned());
        let p = y.argmax().unwrap() as f64;
        if p == t[[i, 0]] {
            accuracy_cnt += 1;
        }
        println!(
            "[{}/{}] (y, target): ({}, {}) Accuracy: {:.2}%",
            i + 1,
            x.nrows(),
            p,
            t[[i, 0]],
            accuracy_cnt as f64 / (i + 1) as f64 * 100.
        );
    }
    println!(
        "Accuracy: {:.2}%",
        accuracy_cnt as f64 / x.nrows() as f64 * 100.
    );
}
