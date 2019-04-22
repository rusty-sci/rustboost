extern crate num;

use std::str::FromStr;
use std::fmt::Debug;

use self::num::{ToPrimitive, Num, FromPrimitive, Float};
use super::super::types::{ScoreType};

#[derive(Debug)]
pub enum NodeType {
  Root, Decision, Leaf 
}

/**
 * Node structure.
 * @property samples : Total number of samples for current node.
 * @property fs_idx : Split feature's index, max index = data dimension.
 * @property fs_val : Split feature's value, equal to avarage value
 *  between two support samples.
 * @property ntype : Node's type.
 * @property score : Node's score metric.
 * @property lchild : Node's left child.
 * @property rchild : Node's right child.
 * @property cl_count : Count of every class instances.
 */
#[derive(Debug)]
pub struct Node<DType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Float + Debug> {
  pub samples: usize,
  pub fs_idx: Option<usize>,
  pub fs_val: Option<DType>,
  pub depth: usize,
  pub ntype: NodeType,
  pub score: ScoreType,
  pub cl_count: Option<Vec<usize>>,
  pub lchild: Option<Box<Node<DType>>>,
  pub rchild: Option<Box<Node<DType>>>,
}

/**
 * Creating and initializing node structure.
 */
impl<DType> Node<DType>
  where DType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Float + Debug {
  pub fn new(ntype: NodeType) -> Self {
    Node {
      samples: 0, depth: 0, ntype: ntype,
      score: 0., fs_idx: None, fs_val: None,
      lchild: None, rchild: None, cl_count: None
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
}

/**
 * Node methods.
 */
impl<DType> Node<DType>
  where DType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Float + Debug {
  pub fn to_leaf(&mut self) {
    self.ntype = NodeType::Leaf;
  }
}