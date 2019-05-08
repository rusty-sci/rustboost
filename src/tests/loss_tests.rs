extern crate rand;

use std::collections::HashSet;
use std::iter::FromIterator;
use rand::Rng;
use super::super::loss::mse::MSE;
use super::super::loss::gini::Gini;
use super::super::types::*;
use super::super::utils::{almost_eq, EPS};

/// Generating data for regression.
/**
 * @param n: Total number of samples.
 * @param m: Dimension of each sample,
 *           first value is target variable,
 *           real dimension of each sample = (m - 1).
 * @return: Generate dataset which is matrix (n x m),
 *          where first column of this matrix is target
 *          values.
 */
fn gen_data(n: usize, m: usize) -> Vec<Vec<dtype>> {
  const MIN: dtype = 10.;
  const MAX: dtype = 20.;
  let mut data: Vec<Vec<dtype>> = Vec::new();
  let mut rng = rand::thread_rng();
  for _ in 0..n {
    let mut sample: Vec<dtype> = Vec::new();
    for _ in 0..m {
      sample.push(rng.gen_range(MIN, MAX))
    }
    data.push(sample);
  }
  return data;
}

/// Simple MSE (Mean Squared Error) computation
/// is used for testing fast MSE computation.
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

/// Testing of fast mse computation
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
  // To MSE we pass 0 as split index so we got: left split size = 0,
  // right split size = length of the data.
  let mse_res = MSE::new(&data, 0).score;
  almost_eq(mse_res, mse_test, EPS);
}

/// Check sliding MSE computation.
/// We slide through the dataset and every iteretion change left and right
/// split size. Left split is increasing in size, while right split is decreasing in size.
#[test]
fn test_sliding_mse() {
  let data: Vec<Vec<dtype>> = gen_data(100, 4);
  let mse_test = mse(&data);
  let mut mse_res = MSE::new(&data, 0);
  almost_eq(mse_test, mse_res.score, EPS);
  for split_idx in 1..data.len() {
    let left_mse_test = mse(&data[0..split_idx]);
    let right_mse_test = mse(&data[split_idx..data.len()]);
    let mse_test = left_mse_test + right_mse_test;
    mse_res.update(&data, split_idx);
    almost_eq(mse_test, mse_res.score, EPS);
  }
}

fn get_classif_data() -> (Vec<Vec<dtype>>, HashSet<usize>)  {
  let data = vec![vec![0.0, 5.1, 3.5, 1.4, 0.2],
                  vec![0.0, 4.9, 3.0, 1.4, 0.2],
                  vec![0.0, 4.7, 3.2, 1.3, 0.2],
                  vec![1.0, 4.6, 3.1, 1.5, 0.2],
                  vec![1.0, 5.0, 3.6, 1.4, 0.2],
                  vec![1.0, 5.0, 3.6, 1.4, 0.2],
                  vec![1.0, 5.0, 3.6, 1.4, 0.2],
                  vec![2.0, 5.0, 3.6, 1.4, 0.2],
                  vec![2.0, 5.0, 3.6, 1.4, 0.2],
                  vec![2.0, 5.0, 3.6, 1.4, 0.2],
                  vec![2.0, 5.0, 3.6, 1.4, 0.2],
                  vec![2.0, 5.0, 3.6, 1.4, 0.2],
                  vec![2.0, 5.4, 3.9, 1.7, 0.4]];
  let classes = HashSet::from_iter(vec![0, 1, 2].iter().cloned());
  return (data, classes);
}

#[test]
fn test_gini_impurity_score() {
  let (data, classes) = get_classif_data();
  let gini = Gini::impurity_score(&data, &classes, data.len());
  let prob_cl0 = 3. / 13.;
  let prob_cl1 = 4. / 13.;
  let prob_cl2 = 6. / 13.;
  let gini_score: dtype = 1. - (prob_cl0 * prob_cl0 + prob_cl1 * prob_cl1 + prob_cl2 * prob_cl2);
  println!("{:?}", gini_score);
  // println!("{:?}", gini.score);
  println!("{:?}", gini);
  // println!("{:?}", gini.right_classes_count);
}