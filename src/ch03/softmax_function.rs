use ndarray::{Array, Dimension};
use ndarray_stats::QuantileExt;

pub fn softmax<D>(x: &Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    let c = *x.max().unwrap(); // Optimize overflow
    // println!("{c}");
    let exp_x = x.mapv(|x| (x - c).exp());
    exp_x.clone() / exp_x.sum()
}
