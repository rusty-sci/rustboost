// TODO: use error struct
// TODO: make stochastic choosing of features for every node
// TODO: correct use of min node samples

use std::f64::MAX;
use std::cmp::Ordering;

use super::{Tree};
use super::node::{Node, NodeType};
use super::super::types::*;
use super::super::config::{LearningTask};
use super::super::loss::mse::MSE;
use super::super::loss::gini::Gini;
use super::super::loss::Loss;
use super::super::utils::{almost_eq, EPS};

impl Tree {

  pub fn cart(&mut self) {
    let mut root: Node = Node::new(NodeType::Root)
      .samples(self.dataset.size)
      .depth(0);
    let data = &mut (*self.dataset.data.borrow_mut())[..];
    self.grow_tree(&mut root, data);
    self.root = Some(root);
  }


  fn grow_tree(&self, node: &mut Node, data: &mut [Vec<dtype>]) {
    let is_root: bool = node.ntype == NodeType::Root;
    let mut lchild: Node = Node::new(NodeType::Decision)
      .depth(node.depth + 1);
    let mut rchild: Node = Node::new(NodeType::Decision)
      .depth(node.depth + 1);

    let mut best_score: dtype = MAX;
    let mut best_f_idx: usize = 0;
    let mut best_s_idx: usize = 0;

    for feature_idx in 1..self.dataset.data_dim {
      Tree::sort_data_by_feature(data, feature_idx);
      let new_loss: Box<Loss> = match self.learning_task {
        LearningTask::Regression => Box::new(MSE::new(data, 0)),
        LearningTask::Classification => {
          Box::new(Gini::new(data, &self.dataset.get_classes_idxs().unwrap()))
        }
      };
      if node.loss.is_none() {
        node.score = new_loss.get_score();
        node.classes_count = new_loss.get_classes_count();
        if (node.score == 0. || node.depth == self.max_tree_depth)
          && !is_root  {
          node.to_leaf();
          return;
        }
      }
      almost_eq(node.score, new_loss.get_score(), EPS);
      node.loss = Some(new_loss);
      for split_idx in self.min_node_samples..data.len() -
        self.min_node_samples {
        node.loss.as_mut().unwrap().update(data, split_idx);
        if node.loss.as_ref().unwrap().get_score() < best_score {
          best_score = node.loss.as_ref().unwrap().get_score();
          best_f_idx = feature_idx;
          best_s_idx = split_idx;
          node.feature_value = Some(data[split_idx][feature_idx]);
        }
      }
    }
    node.loss = None;
    node.feature = Some(best_f_idx);
    Tree::sort_data_by_feature(data, best_f_idx);
    lchild.samples = *(&data[0..best_s_idx].len());
    rchild.samples = *(&data[best_s_idx..node.samples].len());
    node.lchild = Some(Box::new(lchild));
    node.rchild = Some(Box::new(rchild));

    self.grow_tree(node.lchild.as_mut().unwrap(),
      &mut data[0..best_s_idx]);
    self.grow_tree(node.rchild.as_mut().unwrap(),
      &mut data[best_s_idx..node.samples]);
  }


  fn sort_data_by_feature(data: &mut [Vec<dtype>], feature: usize) {
    data.sort_unstable_by(|a, b| {
      match b[feature].partial_cmp(&a[feature]).unwrap() {
        Ordering::Equal => b[0].partial_cmp(&a[0]).unwrap(),
        other => other
      }
    });
  }

}