use std::rc::Rc;

use ch03::{neuralnet_mnist, neuralnet_mnist_batch};
use ndarray::{Array, Array2, Axis, Dimension, array};
use ndarray_rand::{RandomExt, SamplingStrategy};

use crate::{
    ch03::mnist_dataset::{MnistDataset, load_mnist},
    ch04::{
        cross_entropy_error::cross_entropy_error,
        draw::{DrawContent, draw_graph},
        gradient::{function_1, function_2, gradient_descent, numerical_diff, numerical_gradient},
        gradient_simplenet::SimpleNet,
        sum_squares_error::sum_squares_error,
    },
};

mod ch02;
mod ch03;
mod ch04;

fn main() {
    // neuralnet_mnist_batch::run();
    // neuralnet_mnist::run();
    // let y = array![0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0];
    // let t = array![0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
    // let cee = cross_entropy_error(&y, &t);
    // println!("{cee}");
    let MnistDataset {
        x_train_2d,
        t_train,
        x_test_2d,
        t_test,
        ..
    } = load_mnist((60_000, 0, 10_000), true, true);

    // let train_size = x_train_2d.shape()[0];
    // let batch_size = 10;
    // let x_batch = x_train_2d.sample_axis(Axis(0), batch_size, SamplingStrategy::WithoutReplacement);
    // let t_batch = t_train.sample_axis(Axis(0), batch_size, SamplingStrategy::WithoutReplacement);
    // println!("{:?}\n{:?}", x_batch, t_batch);

    // let draw_content = DrawContent {
    //     function: function_1,
    //     caption: String::from("function_1"),
    // };
    // draw_graph(draw_content);

    let net = SimpleNet::new();
    println!("{:?}", net.w);
    let x = array![0.6, 0.9];
    let p = net.predict(&x);
    println!("{:?}", p);
    let t = array![0., 0., 1.];
    let loss = net.loss(&x, &t);
    println!("{:?}", loss);
    let f = |w: &Array2<f64>| net.clone().loss_with_weights(w, &x, &t);
    let dw = numerical_gradient(f, &net.w);
    println!("{:?}", dw);
}
