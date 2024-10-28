use burn::prelude::*;

pub struct Cifar10Batcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Cifar10Batcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}
