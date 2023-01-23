use {
    std::{fs::File, io::BufReader},
    ndarray_npy::{NpzReader, read_npy},
    tch::{
        nn::{self, VarStore, Conv2D, Linear, ConvConfig, BatchNorm, OptimizerConfig},
        data::Iter2,
        Device, 
        Tensor,
        Kind::Float
    },
    npyz::npz::NpzArchive,
};

pub struct Net {
    vs: VarStore,

    conv1: Conv2D,
    conv1_batchnorm: BatchNorm,
    linear1: Linear,
}

impl Net {
    pub fn new() -> Self {
        let vs = VarStore::new(Device::cuda_if_available());
        let root = vs.root();

        let conv1 = nn::conv2d(&root, 28, 28, 3, ConvConfig {
            padding: 1,
            ..Default::default()
        });
        let conv1_batchnorm = nn::batch_norm2d(&root, 256, Default::default());

        let linear1 = nn::linear(&root, 256, 10, Default::default());

        Self {
            vs,

            conv1,
            conv1_batchnorm,
            linear1,
        }
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Tensor {
        xs
            .apply(&self.conv1)
            .apply_t(&self.conv1_batchnorm, train)
            .relu()
            .apply(&self.linear1)
            .softmax(0, Float)  
    }

    fn xs_dataset_from_npz(&self, npz: &mut NpzArchive<BufReader<File>>, name: &str) -> Tensor {
        let npy = npz.by_name(name).unwrap().unwrap();
    
        let shape = npy.shape().to_vec();
        self.dataset_to_tensor(
            npy.data().unwrap().map(|v| v.unwrap()).collect(), 
            shape.iter().map(|v| *v as i64).collect(),
        )
    }

    fn ys_dataset_from_npz(&self, npz: &mut NpzArchive<BufReader<File>>, name: &str) -> Tensor {
        let npy = npz.by_name(name).unwrap().unwrap();

        // perform one-hot encoding
        let mut data = Vec::new();
        let mut len = 0;

        for entry in npy.data().unwrap() {
            let v: u8 = entry.unwrap();
            let mut row = vec![0.0 as f32; 10];
            row[v as usize] = 1.0;
            data.append(&mut row);

            len += 1;
        }

        self.dataset_to_tensor(data, vec![len, 10])
    }

    pub fn dataset_to_tensor(&self, data: Vec<f32>, shape: Vec<i64>) -> Tensor {
        Tensor::of_slice(&data)
            .reshape(&shape)
            .to_device(self.vs.device())
    }
}

fn main() {
    let net = Net::new();

    let mut npz = npyz::npz::NpzArchive::open("./mnist.npz").unwrap();
    let x_train = net.xs_dataset_from_npz(&mut npz, "x_train");
    let y_train = net.ys_dataset_from_npz(&mut npz, "y_train");

    let x_test = net.xs_dataset_from_npz(&mut npz, "x_test");
    let y_test = net.ys_dataset_from_npz(&mut npz, "y_test");

    let mut opt = nn::Adam::default().build(&net.vs, 1e-4).unwrap();

    for epoch in 0..100 {
        println!("running epoch {}", epoch);

        let mut iter = Iter2::new(&x_train, &y_train, 1024);
        iter.shuffle();
        iter.return_smaller_last_batch();

        while let Some((xs, ys)) = iter.next() {
            let loss = net
                .forward(&xs, true)
                .cross_entropy_for_logits(&ys);

            opt.backward_step(&loss);
        }
        
        let test_acc = net
            .forward(&x_test, false)
            .accuracy_for_logits(&y_test);

        println!("epoch {}, test accurracy: {}", epoch, f32::from(test_acc));
    }
}