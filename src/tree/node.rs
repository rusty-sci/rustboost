use super::super::types::*;

#[derive(Debug)]
pub enum NodeType {
  Root, Decision, Leaf 
}

#[derive(Debug)]
pub struct Node {
  pub samples: usize,
  pub fs_idx: Option<usize>,
  pub fs_val: Option<dtype>,
  pub depth: usize,
  pub ntype: NodeType,
  pub score: dtype,
  pub cl_count: Option<Vec<usize>>,
  pub lchild: Option<Box<Node>>,
  pub rchild: Option<Box<Node>>
}

impl Node {

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


// impl<DType> Node<DType>
//   where DType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Float + Debug {
//   pub fn to_leaf(&mut self) {
//     self.ntype = NodeType::Leaf;
//   }
// }