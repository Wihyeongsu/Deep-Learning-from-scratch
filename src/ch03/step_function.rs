use ndarray::{Array, ArrayBase, Data, Dimension};

pub fn step_function<D>(x: &Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}
