use ndarray::Dimension;

use crate::common::bigfloat_array::BigFloatArray;

pub fn identity_function<D: Dimension>(x: &BigFloatArray<D>) -> BigFloatArray<D> {
    x.clone()
}
