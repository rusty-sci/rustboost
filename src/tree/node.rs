use super::super::types::*;
use super::super::loss::Loss;

#[derive(Debug)]
#[derive(PartialEq)]
pub enum NodeType {
  Root, Decision, Leaf 
}

#[derive(Debug)]
pub struct Node {
  pub samples: usize,
  pub feature: Option<usize>,
  pub feature_value: Option<dtype>,
  pub depth: usize,
  pub ntype: NodeType,
  pub score: dtype,
  pub classes_count: Option<Vec<usize>>,
  pub lchild: Option<Box<Node>>,
  pub rchild: Option<Box<Node>>,
  pub loss: Option<Box<Loss>>
}

impl Node {

  pub fn new(ntype: NodeType) -> Self {
    Node {
      samples: 0, depth: 0, ntype: ntype,
      score: 0., feature: None, feature_value: None,
      lchild: None, rchild: None, classes_count: None,
      loss: None
    }
  }


  pub fn samples(self, samples: usize) -> Self {
    Self {
      samples,
      ..self
    }
  }


  pub fn depth(self, depth: usize) -> Self {
    Self {
      depth,
      ..self
    }
  }


  pub fn to_leaf(&mut self) {
    self.ntype = NodeType::Leaf;
  }

}