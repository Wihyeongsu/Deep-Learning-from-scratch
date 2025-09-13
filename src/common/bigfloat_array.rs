use std::ops::{Add, Deref, DerefMut};

use ndarray::{Array, Dimension, ScalarOperand};
use num_bigfloat::BigFloat;

#[derive(Clone, Debug)]
pub struct BigFloatArray<D: Dimension>(pub Array<BigFloat, D>);

impl<D: Dimension> BigFloatArray<D> {}

impl<D: Dimension> Deref for BigFloatArray<D> {
    type Target = Array<BigFloat, D>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<D: Dimension> DerefMut for BigFloatArray<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<D: Dimension + 'static> ScalarOperand for BigFloatArray<D> {}
// Todo: BigFloatArray + float
// Todo: &BigFloatArray + float
// Todo: -BigFloatArray
// Todo: -&BigFloatArray

// BigFloatArray + BigFloat
impl<D: Dimension> Add<BigFloat> for BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn add(self, rhs: BigFloat) -> Self::Output {
        BigFloatArray(self.0.mapv(|x| x + rhs))
    }
}

// &BigFloatArray + &BigFloat
impl<'a, D: Dimension> Add<&'a BigFloat> for &'a BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn add(self, rhs: &'a BigFloat) -> Self::Output {
        BigFloatArray(self.0.mapv(|x| x + rhs))
    }
}

// BigFloatArray + BigFloatArray
impl<D: Dimension> Add<BigFloatArray<D>> for BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn add(self, rhs: BigFloatArray<D>) -> Self::Output {
        if self.0.shape() != rhs.0.shape() {
            panic!(
                "Shape mismatch: {:?} vs {:?}",
                self.0.shape(),
                rhs.0.shape()
            );
        }
        BigFloatArray(self.0 + rhs.0)
    }
}

// &BigFloatArray + &BigFloatArray
impl<'a, 'b, D: Dimension> Add<&'b BigFloatArray<D>> for &'a BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn add(self, rhs: &'b BigFloatArray<D>) -> Self::Output {
        if self.0.shape() != rhs.0.shape() {
            panic!(
                "Shape mismatch: {:?} vs {:?}",
                self.0.shape(),
                rhs.0.shape()
            );
        }
        BigFloatArray(self.0.clone() + rhs.0.clone())
    }
}

// Array<BigFloat, D> -> BigFloatArray<D>
impl<D: Dimension> From<Array<BigFloat, D>> for BigFloatArray<D> {
    fn from(array: Array<BigFloat, D>) -> Self {
        BigFloatArray(array)
    }
}

// BigFloatArray<D> -> Array<BigFloat, D>
impl<D: Dimension> From<BigFloatArray<D>> for Array<BigFloat, D> {
    fn from(wrapper: BigFloatArray<D>) -> Self {
        wrapper.0
    }
}

// Array<f64, D> -> BigFloatArray<D>
impl<D: Dimension> From<Array<f64, D>> for BigFloatArray<D> {
    fn from(array: Array<f64, D>) -> Self {
        BigFloatArray(array.mapv(|x| BigFloat::from(x)))
    }
}

// BigFloatArray<D> -> Array<f64, D>
impl<D: Dimension> From<BigFloatArray<D>> for Array<f64, D> {
    fn from(wrapper: BigFloatArray<D>) -> Self {
        wrapper.0.mapv(|x| x.to_f64())
    }
}
