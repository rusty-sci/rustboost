pub mod node;
pub mod cart;

use std::rc::Rc;

use super::dataset::Dataset;
use super::config::{LearningTask, DEFAULT_LT};
use self::node::*;

pub struct Tree {
  pub root: Option<Node>,
  pub dataset: Rc<Dataset>,
  pub max_tree_depth: usize,
  pub min_node_samples: usize,
  pub learning_task: LearningTask
}

impl Tree {

  pub fn new(dataset: Rc<Dataset>) -> Self {
    let learning_task = LearningTask::new(DEFAULT_LT);
    Tree {
      root: None,
      dataset: dataset,
      max_tree_depth: 0,
      min_node_samples: 0,
      learning_task: learning_task
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


impl Tree {

  pub fn fit(&mut self) {
    self.cart();
  }

}

impl std::fmt::Debug for Tree {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "Tree {{ max_tree_depth: {}, \
    min_node_samples: {:?}, learning_task: {:#?}, root: {:#?} }}",
    self.max_tree_depth, self.min_node_samples,
    self.learning_task, self.root)
  }
}