use ndarray::{Array, ArrayBase, Data, Dimension};

pub fn sigmoid<D>(x: &Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    1. / (1. + (-x).exp())
}
