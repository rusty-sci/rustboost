// TODO: make safe unwrap, use error struct
// TODO: make stochastic choosing of features for every node
extern crate num;

use std::f64::MAX;
use std::collections::HashSet;
use std::str::FromStr;
use std::fmt::Debug;
use std::cmp::Ordering;

use self::num::{ToPrimitive, Num, FromPrimitive, Float};
use super::{Tree};
use super::node::{Node, NodeType};
use super::super::types::{ScoreType};
use super::super::config::{LearningTask};

/**
 * Trait for Tree structure that implements CART algorithm.
 */
impl<TargetType, DType> Tree<TargetType, DType>
  where TargetType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Debug + PartialOrd,
        DType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Float + Debug {

  /**
   * Main CART algorithm method.
   * Performs regression and classification tasks.
   */
  pub fn cart(&mut self) {
    let samples: usize = self.dataset.borrow().data.len();
    // Creating root node.
    let mut root: Node<DType> = Node::<DType>::new(NodeType::Root)
      .samples(samples)
      .depth(0);
    let data_dim: usize = self.dataset.borrow().data_dim;

    // Choose what task to evaluate. Classification or Regression.
    match self.learning_task {

      LearningTask::Classification => {
        // Compute "slow" score (error), at this time Gini Impurity score.
        let result = self.gini_impurity(&self.dataset.borrow().data,
          self.dataset.borrow().classes.as_ref().unwrap(), root.samples);
        root.score = result.0;
        root.cl_count = Some(result.1);
        let number_of_classes: usize = self.dataset.borrow().classes
          .as_ref().unwrap().len();
        let data = &mut (*self.dataset.borrow_mut().data)[..];
        // Start growing tree for classification task.
        self.grow_cart(&mut root, data, data_dim, number_of_classes);
      },

      LearningTask::Regression => {
        root.score = self.mse(&self.dataset.borrow().data).to_f64().unwrap() as ScoreType;
        let data = &mut (*self.dataset.borrow_mut().data)[..];
        self.grow_cart_reg(&mut root, data, data_dim);
      }

    }
    println!("{:#?}", root);
  }

  fn grow_cart_reg(&self, node: &mut Node<DType>, data: &mut [(TargetType, Vec<DType>)],
    data_dim: usize) {
      let data_len = data.len();
      let mut error: ScoreType = MAX;
      let mut best_split_f_idx: usize = 0;
      let mut best_split_s_idx: usize = 0;

      if node.score == 0. || node.depth + 1 > self.max_tree_depth {
        node.to_leaf();
        return;
      }

      let mut lchild: Node<DType> = Node::<DType>::new(NodeType::Decision)
        .depth(node.depth + 1);
      let mut rchild: Node<DType> = Node::<DType>::new(NodeType::Decision)
        .depth(node.depth + 1);

      // println!("{:?}", data);
      // Start checking features
      for f_idx in 0..data_dim {
        // println!("==============================");
        println!("{}", f_idx);
        // println!("{:?}", data[0]);
        data.sort_unstable_by(|a, b| {
          match b.1[f_idx].partial_cmp(&a.1[f_idx]).unwrap() {
            Ordering::Equal => b.0.partial_cmp(&a.0).unwrap(),
            other => other
          }
        });
        let mut left_mse_coefs = self.mse_coefs(&data[0..1]);
        let mut right_mse_coefs = self.mse_coefs(&data[1..data_len]);

        // Spliting by feature value
        for split_idx in 1..data_len {
          // println!("----");
          // println!("{}", split_idx);
          let left_len: TargetType = FromPrimitive::from_usize(data[0..split_idx].len()).unwrap();
          let right_len: TargetType = FromPrimitive::from_usize(data[split_idx..data_len].len())
            .unwrap();
          let left_mean = left_mse_coefs.1 / left_len;
          let right_mean = right_mse_coefs.1 / right_len;
          let left_mse: TargetType = self.fast_mse(left_len, left_mse_coefs.0,
            left_mse_coefs.1, left_mean);
          let right_mse: TargetType = self.fast_mse(right_len, right_mse_coefs.0,
            right_mse_coefs.1, right_mean);

          // println!("left_mean {:?}", left_mean);
          // println!("right_mean {:?}", right_mean);
          // println!("left_mse {:?}", left_mse);
          // println!("right_mse {:?}", right_mse);

          let l_coef = left_len / FromPrimitive::from_usize(data_len).unwrap();
          let r_coef = right_len / FromPrimitive::from_usize(data_len).unwrap();

          let mse = l_coef * left_mse + r_coef * right_mse;

          if (mse.to_f64().unwrap() as ScoreType) < error {
            best_split_f_idx = f_idx;
            best_split_s_idx = split_idx;
            error = mse.to_f64().unwrap() as ScoreType;
          }
          
          // println!("-----------");
          // println!("{:?}", right_mean);
          // println!("{:?}", left_mean);
          // println!("rmse: {:?}", right_mse);
          // println!("lmse: {:?}", left_mse);
          // break;
          // left_sum = left_sum + data[split_idx].0;
          // right_sum = right_sum - data[split_idx].0;
          left_mse_coefs.0 = left_mse_coefs.0 + data[split_idx].0 * data[split_idx].0;
          left_mse_coefs.1 = left_mse_coefs.1 + data[split_idx].0;
          right_mse_coefs.0 = right_mse_coefs.0 - data[split_idx].0 * data[split_idx].0;
          right_mse_coefs.1 = right_mse_coefs.1 - data[split_idx].0;
          // let left_mean = left_sum / FromPrimitive::from_usize(split_idx + 1).unwrap();
          // let right_mean = right_sum / FromPrimitive::from_usize(data_len - split_idx).unwrap();
          // let diff = data[split_idx].0 - left_mean;
          // let right_mean = left_sum / FromPrimitive::from_usize(split_idx + 1).unwrap();
          // let right_diff = data[split_idx].0 - left_sum / FromPrimitive::from_usize(split_idx + 1).unwrap();
          // let right_mean= = self.mean(&data[0..split_idx]);
        }
        // break;
      }

      node.fs_idx = Some(best_split_f_idx);
      lchild.samples = *(&data[0..best_split_s_idx].len());
      rchild.samples = *(&data[best_split_s_idx..node.samples].len());
      node.lchild = Some(Box::new(lchild));
      node.rchild = Some(Box::new(rchild));

      // data.sort_by(|a, b| b.1[best_split_f_idx]
      //   .partial_cmp(&a.1[best_split_f_idx]).unwrap());
      data.sort_unstable_by(|a, b| {
        match b.1[best_split_f_idx].partial_cmp(&a.1[best_split_f_idx]).unwrap() {
          Ordering::Equal => b.0.partial_cmp(&a.0).unwrap(),
          other => other
        }
      });
      
      // self.grow_cart_reg(&mut root, data, data_dim);
      self.grow_cart_reg(node.lchild.as_mut().unwrap(),
        &mut data[0..best_split_s_idx], data_dim);
      self.grow_cart_reg(node.rchild.as_mut().unwrap(),
        &mut data[best_split_s_idx..node.samples], data_dim);
      println!("best error: {:?}", error);
      println!("best f_idx: {:?}", best_split_f_idx);
      println!("best s_idx: {:?}", best_split_s_idx);

      
  }

  fn mse_coefs(&self, data: &[(TargetType, Vec<DType>)]) -> (TargetType, TargetType) {
    let mut target_sq_sum: TargetType = FromPrimitive::from_f64(0.).unwrap();
    let mut target_sum: TargetType = FromPrimitive::from_f64(0.).unwrap();
    for row in data {
      let target = row.0;
      target_sq_sum = target_sq_sum + target * target;
      target_sum = target_sum + target;
    }
    return (target_sq_sum, target_sum)
  }

  fn fast_mse(&self, data_size: TargetType, target_sq_sum: TargetType, target_sum: TargetType,
    mean: TargetType) -> TargetType {
    let multiplier: TargetType = FromPrimitive::from_usize(2).unwrap();
    return target_sq_sum - multiplier*mean*target_sum +
      mean*mean*data_size;
  }

  fn mse(&self, data: &Vec<(TargetType, Vec<DType>)>) -> TargetType {
    let mut sum: TargetType = FromPrimitive::from_f64(0.).unwrap();
    let mut mse: TargetType = FromPrimitive::from_f64(0.).unwrap();
    for row in data {
      sum = sum + row.0;
    }
    let mean = sum / FromPrimitive::from_usize(data.len()).unwrap();
    for row in data {
      mse = mse + (row.0 - mean) * (row.0 - mean);
    }
    return mse / FromPrimitive::from_usize(data.len()).unwrap();
  }

  // fn sum(&self, data: &[(TargetType, Vec<DType>)]) -> TargetType {
  //   let mut sum: TargetType = FromPrimitive::from_f64(0.).unwrap();
  //   for row in data {
  //     sum = sum + row.0;
  //   }
  //   return sum;
  // }

  /**
   * @param node :
   * @param data :
   */
  fn grow_cart(&self, node: &mut Node<DType>, data: &mut [(TargetType, Vec<DType>)],
    data_dim: usize, number_of_classes: usize) {
    // Left node class's proportions.
    let l_class_count_origin: Vec<usize> = vec![0; number_of_classes];

    // Calculate score for Decision of Leaf node using fast gini.
    match node.ntype {
      NodeType::Decision | NodeType::Leaf => {
        let size = node.samples;
        let result = self.fast_gini(&l_class_count_origin, node.cl_count.as_ref().unwrap(),
          0, size, size);
        node.score = result;
      }
      NodeType::Root => ()
    }

    if node.score == 0. || node.depth + 1 > self.max_tree_depth {
      node.to_leaf();
      return;
    }

    let mut lchild: Node<DType> = Node::<DType>::new(NodeType::Decision)
      .depth(node.depth + 1);
    let mut rchild: Node<DType> = Node::<DType>::new(NodeType::Decision)
      .depth(node.depth + 1);

    // Best gini score.
    let mut best_score: ScoreType = MAX;

    // Best feature index, i.e. best dimension to split.
    let mut best_split_f_idx: usize = 0;

    // Best sample index
    let mut best_split_s_idx: usize = 0;

    println!("{:?}", data_dim);
    println!("{:?}", data[0]);
    for f_idx in 0..data_dim {
      println!("{}", f_idx);

      let mut l_class_count = l_class_count_origin.clone();
      let mut r_class_count = node.cl_count.as_ref().unwrap().clone();
      
      //Sort data by feature at index (dimension) = f_idx
      println!("{:?}", data[0]);
      data.sort_unstable_by(|a, b| {
        match b.1[f_idx].partial_cmp(&a.1[f_idx]).unwrap() {
          Ordering::Equal => b.0.partial_cmp(&a.0).unwrap(),
          other => other
        }
      });
      // data.sort_by(|a, b| b.1[f_idx]
      //   .partial_cmp(&a.1[f_idx]).unwrap());

      for split_idx in 1..data.len() {


        let class = *(&data[0..split_idx].last().unwrap().0.to_usize().unwrap());

        l_class_count[class] += 1;
        r_class_count[class] -= 1;

        let left_split_len = *(&data[0..split_idx].len());
        let right_split_len = *(&data[split_idx..data.len()].len());
        let split_gini = self.fast_gini(&l_class_count, &r_class_count, left_split_len,
           right_split_len, node.samples);

        // TODO: need more inteligent choosing of decision boundary
        // Now it uses the last
        if split_gini <= best_score {
          rchild.cl_count = Some(r_class_count.clone());
          lchild.cl_count = Some(l_class_count.clone());
          best_score = split_gini;
          best_split_s_idx = split_idx;
          best_split_f_idx = f_idx;
          node.fs_val = Some((data[split_idx-1].1[f_idx] + data[split_idx].1[f_idx])
            / FromPrimitive::from_f64(2.).unwrap());
        }
      }
    }

    node.fs_idx = Some(best_split_f_idx);
    lchild.samples = *(&data[0..best_split_s_idx].len());
    rchild.samples = *(&data[best_split_s_idx..node.samples].len());
    node.lchild = Some(Box::new(lchild));
    node.rchild = Some(Box::new(rchild));

    // data.sort_by(|a, b| b.1[best_split_f_idx]
    //   .partial_cmp(&a.1[best_split_f_idx]).unwrap());
    data.sort_unstable_by(|a, b| {
      match b.1[best_split_f_idx].partial_cmp(&a.1[best_split_f_idx]).unwrap() {
        Ordering::Equal => b.0.partial_cmp(&a.0).unwrap(),
        other => other
      }
    });
    
    self.grow_cart(node.lchild.as_mut().unwrap(),
      &mut data[0..best_split_s_idx], data_dim,
      number_of_classes);
    self.grow_cart(node.rchild.as_mut().unwrap(),
      &mut data[best_split_s_idx..node.samples], data_dim,
      number_of_classes);
  }

  /**
   * Fast calculation of Gini Impurity score.
   * This function uses previous classes proportions, which were
   * given by gini_impurity at the first step.
   */
  fn fast_gini(&self, l_classes: &Vec<usize>, r_classes: &Vec<usize>, l_len: usize,
    r_len: usize, size: usize) -> ScoreType {

    let gini = |classes: &Vec<usize>, len: usize| {
      let mut score: ScoreType = 0.;
      for class in classes {
        let proportion = *class as ScoreType / len as ScoreType;
        score += proportion * proportion;
      }
      (1. - score) * (len as ScoreType / size as ScoreType)
    };

    if l_len == 0 {
      return gini(r_classes, r_len);
    }

    gini(l_classes, l_len) + gini(r_classes, r_len)
  }

  /**
   * Function for computing Gini Impurity score.
   * Complexity: O(cn), where c - number of classes,
   * n - nuber of samples in data.
   * 
   * This function is only used to compute gini score for root node.
   * Because of complexity O(cn) in subsequent calculations
   * used fast_gini with complexity O(1). 
   * 
   * @param data: dataset.
   * @param classes: uniq classes.
   * @param size: number of samples in dataset.
   * 
   * @return: Function returns tuple, first element of the tuple is Gini Score,
   * second - number of samples belongs to each class.
   */
  fn gini_impurity(&self, data: &Vec<(TargetType, Vec<DType>)>,
    classes: &HashSet<usize>, size: usize) -> (ScoreType, Vec<usize>) {
    let mut score: ScoreType = 0.;
    let mut class_count: Vec<usize> = vec![0; classes.len()];

    for class in classes {
      let mut proportion: ScoreType = 0.;

      for sample in data {
        let sample_class = sample.0.to_usize().unwrap();
        if sample_class == *class {
          proportion += 1.;
        }
      }

      class_count[*class as usize] = proportion as usize;
      proportion /= size as ScoreType;
      score += proportion * proportion;
    }

    ((1. - score), class_count)
  }

}