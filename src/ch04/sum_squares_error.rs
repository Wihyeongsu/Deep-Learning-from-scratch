use ndarray::{Array, Dimension};

pub fn sum_squares_error<D>(y: &Array<f64, D>, t: &Array<f64, D>) -> f64
where
    D: Dimension,
{
    0.5 * ((y - t).pow2()).sum()
}
