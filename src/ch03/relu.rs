use ndarray::{Array, Dimension};

pub fn relu<D>(x: &Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    x.mapv(|x| x.max(0.0))
}
