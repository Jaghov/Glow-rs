use ndarray::Array4;
use ndarray_npz::NpzReader;
use std::fs::{read_dir, File};

pub fn read_npz_file() {
    let mut file_iterator = read_dir("data/images_train_N_5000_T_100_dim_latent_2_dim_obs_2_resolution_32_state_3_sparsity_0.0_net_cosine_seed_24/").unwrap();

    // println!("{:?}", file_iterator.unwrap());
    let mut index: i32 = 0;
    while let Some(sequence) = &file_iterator.next() {
        let seq = sequence.as_ref().unwrap();
        let mut npz =
            NpzReader::new(File::open(seq.path()).expect("should run from project root")).unwrap();
        let a1: Array4<f64> = npz.by_index(0).unwrap();
        index += 1;

        println!("{index}")
    }
}
