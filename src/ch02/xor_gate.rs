use crate::ch02::{and_gate::And, nand_gate::Nand, or_gate::Or};

pub fn Xor(x1:f64, x2:f64) -> f64 {
    let s1 = Nand(x1, x2);
    let s2 = Or(x1, x2);
    let y = And(s1, s2);
    y
}