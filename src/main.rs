mod ch02;
use ndarray::{array};
use ch02::{and_gate::*, nand_gate::*, xor_gate::*};



fn main() {
    println!("{}", Xor(0., 0.));
    println!("{}", Xor(1., 0.));
    println!("{}", Xor(0., 1.));
    println!("{}", Xor(1., 1.));
}
