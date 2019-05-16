use super::super::types::dtype;
use super::Loss;

/// MSE for binary tree regression.
/// score: total loss (left part + right part) of the node.
///   Computes at initialization and after each update() function call.
/// data_len: length of the data.
/// left_coefs: (squared_sum, sum) of the left split.
/// right_coefs: (squared_sum, sum) of the right split.
#[derive(Debug)]
// #[derive(Copy, Clone)]
pub struct MSE {
  score: dtype,
  data_len: usize,
  left_coefs: (dtype, dtype),
  right_coefs: (dtype, dtype)
}

impl Loss for MSE {

  /// Updates left and right coefs for fast mse computation.
  /// And recomputes full mse.
  fn update(&mut self, data: &[Vec<dtype>], split_idx: usize) {
    MSE::update_coefs(&mut self.left_coefs, data, split_idx, |a, b| {a + b});
    MSE::update_coefs(&mut self.right_coefs, data, split_idx, |a, b| {a - b});
    self.compute(split_idx);
  }


  fn get_score(&self) -> dtype {
    self.score
  }

}

/// Public MSE structure methods.
impl MSE {

  pub fn new(data: &[Vec<dtype>], split_idx: usize) -> Self {
    let mut mse = Self {
      score: 0., data_len: data.len(),
      left_coefs: (0., 0.),
      right_coefs: (0., 0.)
    };
    MSE::init_coefs(&mut mse, data, split_idx);
    mse.compute(split_idx);
    mse
  }

}

/// Private methods of MSE structure
impl MSE {

  fn init_coefs(&mut self, data: &[Vec<dtype>], split_idx: usize) {
    self.left_coefs = MSE::coefs(&data[0..split_idx]);
    self.right_coefs = MSE::coefs(&data[split_idx..self.data_len]);
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


  /// Fast MSE computation. O(1)
  fn compute(&mut self, split_idx: usize) {
   self.score = MSE::part_mse(split_idx as dtype, &self.left_coefs) +
      MSE::part_mse((self.data_len - split_idx) as dtype, &self.right_coefs);
  }


  /// Update left and right coefs for fast MSE computation after sliding of split's border.
  fn update_coefs<F: Fn(dtype, dtype) -> dtype>(coefs: &mut (dtype, dtype),
    data: &[Vec<dtype>], split_idx: usize, op: F) {
    coefs.0 = op(coefs.0, data[split_idx - 1][0] * data[split_idx - 1][0]);
    coefs.1 = op(coefs.1, data[split_idx - 1][0]);
  }


  /// Compute MSE for rigth or left split of a binary tree.
  fn part_mse(len: dtype, coefs: &(dtype, dtype)) -> dtype {
    if len > 0. {
      return (coefs.0 - 2. * coefs.1 / len * coefs.1 +
        coefs.1 / len * coefs.1 / len * len) / len;
    } else {
      return 0.;
    }
  }

}