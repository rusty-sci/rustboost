// pub mod cart;

extern crate num;
pub mod node;
pub mod cart;

use std::rc::Rc;
use std::cell::RefCell;
use std::str::FromStr;
use std::fmt::Debug;

use self::num::{ToPrimitive, Num, FromPrimitive, Float};
use super::dataset::Dataset;
use self::node::Node;
// use self::cart::*;
use super::config::{LearningTask, DEFAULT_OBJECTIVE};

/**
 * Tree structure.
 * @property root :
 * @property dataset :
 * @property max_tree_depth :
 * @property min_node_samples :
 */
pub struct Tree <TargetType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Debug + PartialOrd,
  DType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Float + Debug> {
  pub root: Option<Node<DType>>,
  pub dataset: Rc<RefCell<Dataset<TargetType, DType>>>,
  pub max_tree_depth: usize,
  pub min_node_samples: usize,
  pub learning_task: LearningTask
}

/**
 * Creating and initializing tree structure.
 */
impl<TargetType, DType> Tree<TargetType, DType>
  where TargetType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Debug + PartialOrd,
        DType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Float + Debug {

  pub fn new(dataset: Rc<RefCell<Dataset<TargetType, DType>>>) -> Self {
    Tree {
      root: None,
      dataset: dataset,
      max_tree_depth: 0,
      min_node_samples: 0,
      learning_task: LearningTask::new(DEFAULT_OBJECTIVE
        .split(":").collect::<Vec<&str>>()[0])
    }
  }

  pub fn max_tree_depth(self, max_tree_depth: usize) -> Self {
    Self {
      max_tree_depth,
      ..self
    }
  }

  pub fn min_node_samples(self, min_node_samples: usize) -> Self {
    Self {
      min_node_samples,
      ..self
    }
  }

  pub fn learning_task(self, learning_task: LearningTask) -> Self {
    Self {
      learning_task,
      ..self
    }
  }

}

/**
 * Tree methods.
 */
impl<TargetType, DType> Tree<TargetType, DType>
  where TargetType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Debug + PartialOrd,
        DType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Float + Debug {

  pub fn fit(&mut self) {
    self.cart();
  }

}