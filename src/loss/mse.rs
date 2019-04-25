use super::super::types::*;

/// MSE for binary tree regression.
#[derive(Debug)]
pub struct MSE {
  pub loss: dtype,
  data_len: usize,
  left_coefs: (dtype, dtype),
  right_coefs: (dtype, dtype)
}

impl MSE {

  pub fn new(data: &Vec<Vec<dtype>>, split_idx: usize) -> Self {
    let mut mse = Self {
      loss: 0., data_len: data.len(),
      left_coefs: (0., 0.),
      right_coefs: (0., 0.)
    };
    MSE::init_coefs(&mut mse, data, split_idx);
    mse.loss = mse.mse(split_idx);
    mse
  }

  pub fn init_coefs(&mut self, data: &Vec<Vec<dtype>>, split_idx: usize) {
    self.left_coefs = MSE::coefs(&data[0..split_idx]);
    self.right_coefs = MSE::coefs(&data[split_idx..self.data_len]);
  }

  /// Updates left and right coefs for fast mse computation.
  pub fn update(&mut self, data: &Vec<Vec<dtype>>, split_idx: usize) {
    MSE::update_coefs(&mut self.left_coefs, data, split_idx, |a, b| {a + b});
    MSE::update_coefs(&mut self.right_coefs, data, split_idx, |a, b| {a - b});
    self.loss = self.mse(split_idx);
  }

  /// Fast MSE computation. O(1)
  pub fn mse(&self, split_idx: usize) -> dtype {
    let left_len = split_idx as dtype;
    let left_mse = MSE::part_mse(left_len, &self.left_coefs);
    let right_len = (self.data_len - split_idx) as dtype;
    let right_mse = MSE::part_mse(right_len, &self.right_coefs);
    return left_mse + right_mse;
  }

}

/// Private methods of MSE structure
impl MSE {

  fn update_coefs<F: Fn(dtype, dtype) -> dtype>(coefs: &mut (dtype, dtype),
    data: &Vec<Vec<dtype>>, split_idx: usize, op: F) {
    let split_by: usize = split_idx - 1;
    coefs.0 = op(coefs.0, data[split_by][0] * data[split_by][0]);
    coefs.1 = op(coefs.1, data[split_by][0]);
  }

  /// O(n), n - data size
  fn coefs(data: &[Vec<dtype>]) -> (dtype, dtype) {
    let mut target_sq_sum: dtype = 0.;
    let mut target_sum: dtype = 0.;
    for row in data {
      let target = row[0];
      target_sq_sum = target_sq_sum + target * target;
      target_sum = target_sum + target;
    }
    return (target_sq_sum, target_sum)
  }

  /// Compute MSE for rigth or left split of a binary tree.
  fn part_mse(len: dtype, coefs: &(dtype, dtype)) -> dtype {
    if len > 0. {
      let mean: dtype = coefs.1 / len;
      let mse = coefs.0 - 2. * mean * coefs.1 + mean * mean * len;
      return mse / len;
    } else {
      return 0.;
    }
  }
}