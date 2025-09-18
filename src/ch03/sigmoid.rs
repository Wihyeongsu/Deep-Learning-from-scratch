use ndarray::Dimension;

use crate::common::bigfloat_array::BigFloatArray;

pub fn sigmoid<D>(x: &BigFloatArray<D>) -> BigFloatArray<D>
where
    D: Dimension,
{
    let exp = BigFloatArray::from((-x).exp());
    1. / (exp + 1.)
}
