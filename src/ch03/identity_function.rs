use ndarray::{Array, Dimension};

pub fn identity_function<D>(x: &Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    x.clone()
}
