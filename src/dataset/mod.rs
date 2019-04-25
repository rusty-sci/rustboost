extern crate libsvm_parser;

use std::collections::HashSet;
use std::path::Path;

use libsvm_parser::LIBSVMParser;

use super::types::*;

#[derive(Debug)]
pub struct Dataset {
  pub data: Vec<Vec<dtype>>,
  pub data_dim: usize,
  pub classes: Option<HashSet<usize>>,
  pub size: usize
}

impl Dataset {

  pub fn from_libsvm(path: &str, classif: bool) -> Self {
    let parser = LIBSVMParser::new().is_classification(classif);
    let (data, classes) = parser.parse_file::<dtype, usize>(Path::new(path));
    if data.len() == 0 {
      panic!("There is no data.");
    }
    if data[0].len() < 2 {
      panic!("Wrong data.");
    }
    let data_dim: usize = *(&data[0][1..].len());
    let size: usize = data.len();
    Self {
      data: data,
      data_dim: data_dim,
      classes: classes,
      size: size
    }
  }

}