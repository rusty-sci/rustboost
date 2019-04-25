pub mod node;
pub mod cart;

use std::rc::Rc;
use std::cell::RefCell;

use super::dataset::Dataset;
use super::config::{LearningTask, DEFAULT_OBJECTIVE};
use self::node::*;

#[derive(Debug)]
pub struct Tree {
  pub root: Option<Node>,
  pub dataset: Rc<RefCell<Dataset>>,
  pub max_tree_depth: usize,
  pub min_node_samples: usize,
  pub learning_task: LearningTask
}

impl Tree {

  pub fn new(dataset: Rc<RefCell<Dataset>>) -> Self {
    let learning_task = LearningTask::new(DEFAULT_OBJECTIVE
      .split(":").collect::<Vec<&str>>()[0]);
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