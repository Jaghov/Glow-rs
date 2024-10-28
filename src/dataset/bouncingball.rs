use ndarray::Array4;
use ndarray_npz::NpzReader;
use std::fs::File;

pub fn read_npz_file() {
    let mut npz =
        NpzReader::new(File::open("data/00000.npz").expect("should run from project root"))
            .unwrap();
    let a1: Array4<f64> = npz.by_index(0).unwrap();

    println!("{:?}", a1);
}
