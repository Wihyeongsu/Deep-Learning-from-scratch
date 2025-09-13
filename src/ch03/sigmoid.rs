use ndarray::{Array, ArrayBase, Data, Dimension};
use num_bigfloat::BigFloat;

use crate::common::bigfloat_array::BigFloatArray;

pub fn sigmoid<D>(x: &BigFloatArray<D>) -> BigFloatArray<D>
where
    D: Dimension,
{
    1. / ((-x).exp() + 1.)
}
