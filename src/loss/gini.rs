// use std::collections::HashSet;
use super::super::types::dtype;
use super::Loss;

#[derive(Debug)]
// #[derive(Copy, Clone)]
pub struct Gini {
  score: dtype,
  data_len: usize,
  left_classes_count: Vec<usize>,
  right_classes_count: Vec<usize>,
  class_count: Vec<usize>
}

impl Loss for Gini {

  fn update(&mut self, data: &[Vec<dtype>], split_idx: usize) {
    let class = data[split_idx - 1][0] as usize;
    self.left_classes_count[class] += 1;
    self.right_classes_count[class] -= 1;
    self.score = self.fast_impurity(data, split_idx);
  }


  fn get_score(&self) -> dtype {
    self.score
  }

  fn get_classes_count(&self) -> Option<Vec<usize>> {
    Some(self.class_count.clone())
  }

}

impl Gini {

  pub fn new(data: &[Vec<dtype>], classes: &Vec<usize>) -> Self {
    let (score, right_classes_count) = Gini::impurity(data, classes, data.len());
    Self {
      score: score,
      data_len: data.len(),
      class_count: right_classes_count.clone(),
      left_classes_count: vec![0; classes.len()],
      right_classes_count: right_classes_count
    }
  }

}

impl Gini {

  fn impurity(data: &[Vec<dtype>], classes: &Vec<usize>,
    data_len: usize) -> (dtype, Vec<usize>) {
    let mut score: dtype = 0.;
    let mut class_count: Vec<usize> = vec![0; classes.len()];
    for class in classes {
      let mut proportion: dtype = 0.;
      for sample in data {
        let sample_class = sample[0] as usize;
        if sample_class == *class {
          proportion += 1.;
        }
      }
      class_count[*class] = proportion as usize;
      proportion /= data_len as dtype;
      score += proportion * proportion;
    }
    ((1. - score), class_count)
  }


  fn fast_impurity(&self, data: &[Vec<dtype>], split_idx: usize) -> dtype {
    let gini = |classes: &Vec<usize>, len: usize| {
      let mut score: dtype = 0.;
      for class in classes {
        let proportion = *class as dtype / len as dtype;
        score += proportion * proportion;
      }
      //The Gini index for each group must then be weighted by the size of the group,
      //relative to all of the samples in the parent,
      //e.g. all samples that are currently being grouped.
      (1. - score) * (len as dtype / self.data_len as dtype)
    };
    if data[0..split_idx].len() == 0 {
      return gini(&self.right_classes_count, data[split_idx..data.len()].len());
    }
    gini(&self.left_classes_count, data[0..split_idx].len()) +
      gini(&self.right_classes_count, data[split_idx..data.len()].len())
  }

}