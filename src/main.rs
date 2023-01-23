use {
    std::{fs::File, io::BufReader},
    ndarray_npy::{NpzReader, read_npy},
    tch::Tensor,
    npyz::npz::NpzArchive,
};

fn main() {
    println!("Hello, world!");

    let mut npz = npyz::npz::NpzArchive::open("./mnist.npz").unwrap();
    let x_train = x_dataset_from_npz(&mut npz, "x_train");
    let x_test = x_dataset_from_npz(&mut npz, "x_test");
    let y_train = y_dataset_from_npz(&mut npz, "y_train");
    let y_test = y_dataset_from_npz(&mut npz, "y_test");

    println!("result: {:?}", x_test);
}

fn y_dataset_from_npz(npz: &mut NpzArchive<BufReader<File>>, name: &str) -> Vec<Tensor> {
    let npy = npz.by_name(name).unwrap().unwrap();

    let shape = npy.shape().to_vec();
    println!("y shape is: {:?}", shape);

    dataset_to_tensor(
        npy.data().unwrap().map(|v| v.unwrap()).collect(),
        shape[0],
        shape[1]
    )
}

fn x_dataset_from_npz(npz: &mut NpzArchive<BufReader<File>>, name: &str) -> Vec<Tensor> {
    let npy = npz.by_name(name).unwrap().unwrap();

    let shape = npy.shape().to_vec();
    dataset_to_tensor(
        npy.data().unwrap().map(|v| v.unwrap()).collect(), 
        shape[0], 
        shape[1] * shape[2]
    )
}

fn dataset_to_tensor(data: Vec<u8>, total_entries: u64, tensor_len: u64) -> Vec<Tensor> {
    let mut res = Vec::new();
    for i in 0..total_entries {
        let start = (i * tensor_len) as usize;
        let end = start + tensor_len as usize;
        res.push(Tensor::of_slice(&data[start..end]));
    }
    res
}