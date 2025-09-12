use ndarray::{Array, Dimension};

pub fn cross_entropy_error<D>(y: &Array<f64, D>, t: &Array<f64, D>) -> f64
where
    D: Dimension,
{
    let delta = 1e-7; // log(0) = -inf
    let batch_size = y.shape()[0] as f64;
    -(t * (y + delta).ln()).sum() / batch_size
}
