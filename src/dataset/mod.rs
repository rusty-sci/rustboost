extern crate libsvm_parser;

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use libsvm_parser::LIBSVMParser;

use super::types::*;

#[derive(Debug)]
pub struct Dataset {
  pub data: Rc<RefCell<Vec<Vec<dtype>>>>,
  pub data_dim: usize,
  pub classes: Option<Rc<HashMap<String, dtype>>>,
  pub size: usize
}

impl Dataset {

  pub fn from_libsvm(path: &str, classif: bool) -> Self {
    let parser = LIBSVMParser::new().is_classification(classif);
    let (data, classes) = parser.parse_file::<dtype>(Path::new(path));
    if data.len() == 0 {
      panic!("There is no data.");
    }
    if data[0].len() < 2 {
      panic!("Wrong data.");
    }
    let data_dim: usize = *(&data[0][1..].len());
    let size: usize = data.len();
    Self {
      data: Rc::new(RefCell::new(data)),
      data_dim: data_dim,
      classes: match classes {
        Some(classes) => Some(Rc::new(classes)),
        None => None
      },
      size: size
    }
  }


  pub fn get_classes_names(&self) -> Option<Vec<String>> {
    match self.classes {
      Some(ref classes) => {
        Some(classes.iter().map(|(k, _)| k.clone()).collect())
      },
      None => None
    }
  }


  pub fn get_classes_idxs(&self) -> Option<Vec<usize>> {
    match self.classes {
      Some(ref classes) => {
        Some(classes.iter().map(|(_, &cl)| cl as usize).collect())
      },
      None => None
    }
  }

}