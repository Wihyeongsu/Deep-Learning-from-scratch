use ndarray::array;

pub fn And(x1:f64, x2:f64) -> f64 {
    let x = array![x1, x2];
    let w = array![0.5, 0.5];
    let b = -0.7;
    let y = (x*w).sum() + b;
    if y <= 0. {
        return 0.;
    }
    else {
        return 1.;
    }
}