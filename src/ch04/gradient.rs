use ndarray::{Array, Array1, Dimension, array};
use ndarray_rand::rand_distr::num_traits::Pow;
use plotters::element;

pub fn function_1(x: f64) -> f64 {
    0.01 * x.pow(2) + 0.1 * x
}

pub fn function_2(x: &Array1<f64>) -> f64 {
    x[0].pow(2) + x[1].pow(2)
}

pub fn numerical_diff<F>(f: F, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = 1e-4;
    (f(x + h) - f(x - h)) / (2. * h)
}

pub fn numerical_gradient<F, D>(f: F, x: &Array<f64, D>) -> Array<f64, D>
where
    F: Fn(&Array<f64, D>) -> f64,
    D: Dimension,
    D::Pattern: ndarray::NdIndex<D>,
{
    let h = 1e-4;
    let mut grad = Array::zeros(x.raw_dim());

    for (multi_idx, &v) in x.clone().indexed_iter() {
        let mut x_cloned = x.clone();
        // f(x+h)
        x_cloned[multi_idx.clone()] = v + h;
        let fxh1 = f(&x_cloned);

        // f(x-h)
        x_cloned[multi_idx.clone()] = v - h;
        let fxh2 = f(&x_cloned);

        grad[multi_idx.clone()] = (fxh1 - fxh2) / (2. * h);
    }

    grad
}

pub fn gradient_descent<F, D>(
    f: F,
    init_x: &Array<f64, D>,
    lr: Option<f64>,
    step_num: Option<usize>,
) -> (Array<f64, D>, Vec<Array<f64, D>>)
where
    F: Fn(&Array<f64, D>) -> f64 + Copy,
    D: Dimension,
    D::Pattern: ndarray::NdIndex<D>,
{
    let mut x = init_x.clone();
    let lr = lr.unwrap_or(0.01);
    let step_num = step_num.unwrap_or(100);
    let mut x_history = Vec::new();

    for _ in 0..step_num {
        x_history.push(x.clone());

        let grad = numerical_gradient(f, &x);
        x = x - lr * grad;
    }

    (x, x_history)
}
