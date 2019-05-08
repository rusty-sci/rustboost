use std::collections::HashSet;
use super::super::types::dtype;

#[derive(Debug)]
// #[derive(Copy, Clone)]
pub struct Gini {
  pub score: dtype,
  data_len: usize,
  left_classes_count: Vec<usize>,
  right_classes_count: Vec<usize>
}

impl Gini {
  pub fn new(data: &[Vec<dtype>], classes: &HashSet<usize>) -> Self {
    let (score, right_classes_count) = Gini::impurity(data, classes, data.len());
    Self {
      score: score,
      data_len: data.len(),
      left_classes_count: vec![0; classes.len()],
      right_classes_count: right_classes_count
    }
  }

  fn impurity(data: &[Vec<dtype>], classes: &HashSet<usize>,
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

      class_count[*class as usize] = proportion as usize;
      proportion /= data_len as dtype;
      score += proportion * proportion;
    }

    ((1. - score), class_count)
  }

  #[cfg(test)]
  pub fn impurity_score(data: &[Vec<dtype>], classes: &HashSet<usize>,
    data_len: usize) -> (dtype, Vec<usize>) {
      return Gini::impurity(data, classes, data_len);
  }

}


 // /**
  //  * Fast calculation of Gini Impurity score.
  //  * This function uses previous classes proportions, which were
  //  * given by gini_impurity at the first step.
  //  */
  // fn fast_gini(&self, l_classes: &Vec<usize>, r_classes: &Vec<usize>, l_len: usize,
  //   r_len: usize, size: usize) -> ScoreType {

  //   let gini = |classes: &Vec<usize>, len: usize| {
  //     let mut score: ScoreType = 0.;
  //     for class in classes {
  //       let proportion = *class as ScoreType / len as ScoreType;
  //       score += proportion * proportion;
  //     }
  //     (1. - score) * (len as ScoreType / size as ScoreType)
  //   };

  //   if l_len == 0 {
  //     return gini(r_classes, r_len);
  //   }

  //   gini(l_classes, l_len) + gini(r_classes, r_len)
  // }

  // /**
  //  * Function for computing Gini Impurity score.
  //  * Complexity: O(cn), where c - number of classes,
  //  * n - nuber of samples in data.
  //  * 
  //  * This function is only used to compute gini score for root node.
  //  * Because of complexity O(cn) in subsequent calculations
  //  * used fast_gini with complexity O(1). 
  //  * 
  //  * @param data: dataset.
  //  * @param classes: uniq classes.
  //  * @param size: number of samples in dataset.
  //  * 
  //  * @return: Function returns tuple, first element of the tuple is Gini Score,
  //  * second - number of samples belongs to each class.
  //  */
  // fn gini_impurity(&self, data: &Vec<(TargetType, Vec<DType>)>,
  //   classes: &HashSet<usize>, size: usize) -> (ScoreType, Vec<usize>) {
  //   let mut score: ScoreType = 0.;
  //   let mut class_count: Vec<usize> = vec![0; classes.len()];

  //   for class in classes {
  //     let mut proportion: ScoreType = 0.;

  //     for sample in data {
  //       let sample_class = sample.0.to_usize().unwrap();
  //       if sample_class == *class {
  //         proportion += 1.;
  //       }
  //     }

  //     class_count[*class as usize] = proportion as usize;
  //     proportion /= size as ScoreType;
  //     score += proportion * proportion;
  //   }

  //   ((1. - score), class_count)
  // }