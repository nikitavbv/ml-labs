use {
    std::{fs::File, io::BufReader},
    ndarray_npy::{NpzReader, read_npy},
    tch::Tensor,
    npyz::npz::NpzArchive,
};

fn main() {
    println!("Hello, world!");

    /*let mut npz = NpzReader::new(File::open("./mnist.npz").unwrap()).unwrap();
    let names = npz.names().unwrap();

    let x_train: Array4<f32> = npz.by_name("x_train.npy").unwrap();*/

    let mut npz = npyz::npz::NpzArchive::open("./mnist.npz").unwrap();
    let x_test = x_dataset_from_npz(&mut npz, "x-test");

    println!("result: {:?}", x_test);
}

fn x_dataset_from_npz(npz: &mut NpzArchive<BufReader<File>>, name: &str) -> Vec<Tensor> {
    let npy = npz.by_name(name).unwrap().unwrap();

    let shape = npy.shape().to_vec();
    x_dataset(
        npy.data().unwrap().map(|v| v.unwrap()).collect(), 
        shape[0], 
        shape[1] * shape[2]
    )
}

fn x_dataset(data: Vec<u8>, total_entries: u64, tensor_len: u64) -> Vec<Tensor> {
    let mut res = Vec::new();
    for i in 0..total_entries {
        let start = (i * tensor_len) as usize;
        let end = start + tensor_len as usize;
        res.push(Tensor::of_slice(&data[start..end]));
    }
    res
}