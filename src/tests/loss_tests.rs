extern crate rand;

use rand::Rng;
use pretty_assertions::{assert_eq};
use super::super::loss::mse::MSE;
use super::super::types::*;

const EPS: f64 = 10_000_0000f64;

fn gen_data(N: usize, M: usize) -> Vec<Vec<dtype>> {
  const MIN: dtype = 10.;
  const MAX: dtype = 20.;
  let mut data: Vec<Vec<dtype>> = Vec::new();
  let mut rng = rand::thread_rng();
  for _ in 0..N {
    let mut sample: Vec<dtype> = Vec::new();
    for _ in 0..M {
      sample.push(rng.gen_range(MIN, MAX))
    }
    data.push(sample);
  }
  return data;
}

fn mse(data: &[Vec<dtype>]) -> dtype {
  let mut sum: dtype = 0.;
  let mut mse: dtype = 0.;
  for row in data {
    sum = sum + row[0];
  }
  let mean: dtype = sum / data.len() as dtype;
  for row in data {
    mse = mse + (row[0] - mean) * (row[0] - mean);
  }
  return mse / data.len() as dtype;
}

fn almost_eq(v1: dtype, v2: dtype, eps: dtype) {
  assert_eq!((v1 * eps).round() / eps,
    (v2 * eps).round() / eps);
}

#[test]
fn test_mse() {
  // let data = vec![vec![0.0, 5.1, 3.5, 1.4, 0.2],
  //                 vec![2.0, 4.9, 3.0, 1.4, 0.2],
  //                 vec![8.0, 4.7, 3.2, 1.3, 0.2],
  //                 vec![2.0, 4.6, 3.1, 1.5, 0.2]];
                  // vec![1.0, 5.0, 3.6, 1.4, 0.2],
                  // vec![8., 5.4, 3.9, 1.7, 0.4]];
  let data: Vec<Vec<dtype>> = gen_data(100, 4);
  let mse_test = mse(&data);
  let mse_res = MSE::new(&data, 0).loss;
  almost_eq(mse_test, mse_res, EPS);
}

#[test]
fn test_sliding_mse() {
  let data: Vec<Vec<dtype>> = gen_data(100, 4);
  let mse_test = mse(&data);
  let mut mse_res = MSE::new(&data, 0);
  almost_eq(mse_test, mse_res.loss, EPS);
  for split_idx in 1..data.len() {
    let left_mse_test = mse(&data[0..split_idx]);
    let right_mse_test = mse(&data[split_idx..data.len()]);
    let mse_test = left_mse_test + right_mse_test;
    mse_res.update(&data, split_idx);
    almost_eq(mse_test, mse_res.loss, EPS);
  }
}