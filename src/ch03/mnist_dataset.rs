use mnist::{Mnist, MnistBuilder};
use ndarray::{Array2, Array3, Ix2, Ix3};

use crate::common::bigfloat_array::BigFloatArray;

pub struct MnistDataset {
    pub x_train_2d: BigFloatArray<Ix2>,
    pub x_train_3d: BigFloatArray<Ix3>,
    pub t_train: BigFloatArray<Ix2>,

    pub x_val_2d: BigFloatArray<Ix2>,
    pub x_val_3d: BigFloatArray<Ix3>,
    pub t_val: BigFloatArray<Ix2>,

    pub x_test_2d: BigFloatArray<Ix2>,
    pub x_test_3d: BigFloatArray<Ix3>,
    pub t_test: BigFloatArray<Ix2>,
}

pub fn load_mnist(
    (train_length, validation_length, test_length): (u32, u32, u32),
    normalize: bool,
    one_hot_encoding: bool,
) -> MnistDataset {
    let mut mnist_builder = MnistBuilder::new();
    mnist_builder
        .base_path("data/")
        .training_set_length(train_length)
        .validation_set_length(validation_length)
        .test_set_length(test_length);

    if one_hot_encoding {
        mnist_builder.label_format_one_hot();
    }

    let Mnist {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        tst_img,
        tst_lbl,
        ..
    } = mnist_builder.finalize();

    let x_train_2d = Array2::from_shape_vec((train_length as usize, 784), trn_img.clone())
        .expect("Error converting images to Array2 struct")
        .map(|x| {
            if normalize {
                *x as f64 / 256.0
            } else {
                *x as f64
            }
        });

    let x_train_3d = Array3::from_shape_vec((train_length as usize, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| {
            if normalize {
                *x as f64 / 256.0
            } else {
                *x as f64
            }
        });

    let t_train: Array2<f64> = if one_hot_encoding {
        Array2::from_shape_vec((train_length as usize, 10), trn_lbl)
    } else {
        Array2::from_shape_vec((train_length as usize, 1), trn_lbl)
    }
    .expect("Error converting training labels to Array2 struct")
    .map(|x| *x as f64);
    // println!(
    //     "The first digit is a {:?}",
    //     t_train.slice(s![image_num, ..])
    // );

    let x_val_2d = Array2::from_shape_vec((validation_length as usize, 784), val_img.clone())
        .expect("Error converting images to Array2 struct")
        .map(|x| {
            if normalize {
                *x as f64 / 256.0
            } else {
                *x as f64
            }
        });

    let x_val_3d = Array3::from_shape_vec((validation_length as usize, 28, 28), val_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| {
            if normalize {
                *x as f64 / 256.0
            } else {
                *x as f64
            }
        });

    let t_val: Array2<f64> = if one_hot_encoding {
        Array2::from_shape_vec((validation_length as usize, 10), val_lbl)
    } else {
        Array2::from_shape_vec((validation_length as usize, 1), val_lbl)
    }
    .expect("Error converting validation labels to Array2 struct")
    .map(|x| *x as f64);

    let x_test_2d = Array2::from_shape_vec((test_length as usize, 784), tst_img.clone())
        .expect("Error converting images to Array2 struct")
        .map(|x| {
            if normalize {
                *x as f64 / 256.0
            } else {
                *x as f64
            }
        });

    let x_test_3d = Array3::from_shape_vec((test_length as usize, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| {
            if normalize {
                *x as f64 / 256.0
            } else {
                *x as f64
            }
        });

    let t_test: Array2<f64> = if one_hot_encoding {
        Array2::from_shape_vec((test_length as usize, 10), tst_lbl)
    } else {
        Array2::from_shape_vec((test_length as usize, 1), tst_lbl)
    }
    .expect("Error converting testing labels to Array2 struct")
    .map(|x| *x as f64);

    MnistDataset {
        x_train_2d: BigFloatArray::from(x_train_2d),
        x_train_3d: BigFloatArray::from(x_train_3d),
        t_train: BigFloatArray::from(t_train),
        x_val_2d: BigFloatArray::from(x_val_2d),
        x_val_3d: BigFloatArray::from(x_val_3d),
        t_val: BigFloatArray::from(t_val),
        x_test_2d: BigFloatArray::from(x_test_2d),
        x_test_3d: BigFloatArray::from(x_test_3d),
        t_test: BigFloatArray::from(t_test),
    }
}
