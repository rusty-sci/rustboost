pub mod mse;
pub mod gini;

use super::types::dtype;

pub trait Loss {
  fn update(&mut self, data: &[Vec<dtype>], split_idx: usize);
  fn get_score(&self) -> dtype;

  fn get_classes_count(&self) -> Option<Vec<usize>> { None }
}

impl std::fmt::Debug for Loss {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "{}", self.get_score())
  }
}