use std::ops::{Add, Deref, DerefMut, Div, Mul, Neg, Sub};

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

// Add
// BigFloatArray + f64 
impl<D: Dimension> Add<f64> for BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn add(self, rhs: f64) -> Self::Output {
        let big_rhs = BigFloat::from(rhs);
        BigFloatArray(self.0.mapv(|x| x + &big_rhs))
    }
}

// &BigFloatArray + f64 
impl<'a, D: Dimension> Add<f64> for &'a BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn add(self, rhs: f64) -> Self::Output {
        let big_rhs = BigFloat::from(rhs);
        BigFloatArray(self.0.mapv(|x| x + &big_rhs))
    }
}

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

    fn add(self, rhs: Self) -> Self::Output {
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

// =============================
// Neg
// -BigFloatArray
impl<D: Dimension> Neg for BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn neg(self) -> Self::Output {
        BigFloatArray(self.0.mapv(|x| -x))
    }
}

// -&BigFloatArray
impl<'a, D: Dimension> Neg for &'a BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn neg(self) -> Self::Output {
        BigFloatArray(self.0.mapv(|x| -x))
    }
}


// =============================
// Sub
// =============================
// BigFloatArray - f64
impl<D: Dimension> Sub<f64> for BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn sub(self, rhs: f64) -> Self::Output {
        let big_rhs = BigFloat::from(rhs);
        BigFloatArray(self.0.mapv(|x| x - &big_rhs))
    }
}

// &BigFloatArray - f64
impl<'a, D: Dimension> Sub<f64> for &'a BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn sub(self, rhs: f64) -> Self::Output {
        let big_rhs = BigFloat::from(rhs);
        BigFloatArray(self.0.mapv(|x| x - &big_rhs))
    }
}

// BigFloatArray - BigFloat
impl<D: Dimension> Sub<BigFloat> for BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn sub(self, rhs: BigFloat) -> Self::Output {
        BigFloatArray(self.0.mapv(|x| x - &rhs))
    }
}

// &BigFloatArray - &BigFloat
impl<'a, D: Dimension> Sub<&'a BigFloat> for &'a BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn sub(self, rhs: &'a BigFloat) -> Self::Output {
        BigFloatArray(self.0.mapv(|x| x - rhs))
    }
}

// BigFloatArray - BigFloatArray
impl<D: Dimension> Sub<BigFloatArray<D>> for BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn sub(self, rhs: BigFloatArray<D>) -> Self::Output {
        if self.0.shape() != rhs.0.shape() {
            panic!(
                "Shape mismatch: {:?} vs {:?}",
                self.0.shape(),
                rhs.0.shape()
            );
        }
        BigFloatArray(self.0 - rhs.0)
    }
}

// &BigFloatArray - &BigFloatArray
impl<'a, 'b, D: Dimension> Sub<&'b BigFloatArray<D>> for &'a BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn sub(self, rhs: &'b BigFloatArray<D>) -> Self::Output {
        if self.0.shape() != rhs.0.shape() {
            panic!(
                "Shape mismatch: {:?} vs {:?}",
                self.0.shape(),
                rhs.0.shape()
            );
        }
        BigFloatArray(self.0.clone() - rhs.0.clone())
    }
}

// =============================
// Mul
// =============================
// BigFloatArray * f64
impl<D: Dimension> Mul<f64> for BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn mul(self, rhs: f64) -> Self::Output {
        let big_rhs = BigFloat::from(rhs);
        BigFloatArray(self.0.mapv(|x| x * &big_rhs))
    }
}

// &BigFloatArray * f64
impl<'a, D: Dimension> Mul<f64> for &'a BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn mul(self, rhs: f64) -> Self::Output {
        let big_rhs = BigFloat::from(rhs);
        BigFloatArray(self.0.mapv(|x| x * &big_rhs))
    }
}

// BigFloatArray * BigFloat
impl<D: Dimension> Mul<BigFloat> for BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn mul(self, rhs: BigFloat) -> Self::Output {
        BigFloatArray(self.0.mapv(|x| x * &rhs))
    }
}

// &BigFloatArray * &BigFloat
impl<'a, D: Dimension> Mul<&'a BigFloat> for &'a BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn mul(self, rhs: &'a BigFloat) -> Self::Output {
        BigFloatArray(self.0.mapv(|x| x * rhs))
    }
}

// BigFloatArray * BigFloatArray (element-wise)
impl<D: Dimension> Mul<BigFloatArray<D>> for BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn mul(self, rhs: BigFloatArray<D>) -> Self::Output {
        if self.0.shape() != rhs.0.shape() {
            panic!(
                "Shape mismatch: {:?} vs {:?}",
                self.0.shape(),
                rhs.0.shape()
            );
        }
        BigFloatArray(self.0 * rhs.0)
    }
}

// &BigFloatArray * &BigFloatArray (element-wise)
impl<'a, 'b, D: Dimension> Mul<&'b BigFloatArray<D>> for &'a BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn mul(self, rhs: &'b BigFloatArray<D>) -> Self::Output {
        if self.0.shape() != rhs.0.shape() {
            panic!(
                "Shape mismatch: {:?} vs {:?}",
                self.0.shape(),
                rhs.0.shape()
            );
        }
        BigFloatArray(self.0.clone() * rhs.0.clone())
    }
}



// =============================
// Div
// =============================
// BigFloatArray / f64
impl<D: Dimension> Div<f64> for BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn div(self, rhs: f64) -> Self::Output {
        let big_rhs = BigFloat::from(rhs);
        BigFloatArray(self.0.mapv(|x| x / &big_rhs))
    }
}

// &BigFloatArray / f64
impl<'a, D: Dimension> Div<f64> for &'a BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn div(self, rhs: f64) -> Self::Output {
        let big_rhs = BigFloat::from(rhs);
        BigFloatArray(self.0.mapv(|x| x / &big_rhs))
    }
}

// BigFloatArray / BigFloat
impl<D: Dimension> Div<BigFloat> for BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn div(self, rhs: BigFloat) -> Self::Output {
        BigFloatArray(self.0.mapv(|x| x / &rhs))
    }
}

// &BigFloatArray / &BigFloat
impl<'a, D: Dimension> Div<&'a BigFloat> for &'a BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn div(self, rhs: &'a BigFloat) -> Self::Output {
        BigFloatArray(self.0.mapv(|x| x / rhs))
    }
}

// BigFloatArray / BigFloatArray (element-wise)
impl<D: Dimension> Div<BigFloatArray<D>> for BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn div(self, rhs: BigFloatArray<D>) -> Self::Output {
        if self.0.shape() != rhs.0.shape() {
            panic!(
                "Shape mismatch: {:?} vs {:?}",
                self.0.shape(),
                rhs.0.shape()
            );
        }
        BigFloatArray(self.0 / rhs.0)
    }
}

// &BigFloatArray / &BigFloatArray (element-wise)
impl<'a, 'b, D: Dimension> Div<&'b BigFloatArray<D>> for &'a BigFloatArray<D> {
    type Output = BigFloatArray<D>;

    fn div(self, rhs: &'b BigFloatArray<D>) -> Self::Output {
        if self.0.shape() != rhs.0.shape() {
            panic!(
                "Shape mismatch: {:?} vs {:?}",
                self.0.shape(),
                rhs.0.shape()
            );
        }
        BigFloatArray(self.0.clone() / rhs.0.clone())
    }
}

// f64 / BigFloatArray
impl<D: Dimension> Div<BigFloatArray<D>> for f64 {
    type Output = BigFloatArray<D>;

    fn div(self, rhs: BigFloatArray<D>) -> Self::Output {
        let big_lhs = BigFloat::from(self);
        BigFloatArray(rhs.0.mapv(|x| big_lhs / x))
    }
}

// f64 / &BigFloatArray
impl<'a, D: Dimension> Div<&'a BigFloatArray<D>> for f64 {
    type Output = BigFloatArray<D>;

    fn div(self, rhs: &'a BigFloatArray<D>) -> Self::Output {
        let big_lhs = BigFloat::from(self);
        BigFloatArray(rhs.0.mapv(|x| big_lhs / x))
    }
}

// =============================
// From
// =============================
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
