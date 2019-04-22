extern crate num;

mod libsvm_parser;

use std::path::Path;
use std::fs::File;
use std::error::Error;
use std::collections::HashSet;
use std::str::FromStr;
use std::fmt::Debug;

use self::num::{ToPrimitive, Num, FromPrimitive, Float};
use super::config::{LearningTask};
use super::types::{DataType};

/**
 * Dataset structure.
 * @property data : Train data.
 * @property data_dim : Dimension fo every sample in the data.
 * @property classes : Number of classes in dataset, if regression task = None.
 */
pub struct Dataset<TargetType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Debug,
  DType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Float + Debug> {
  pub data: DataType<TargetType, DType>,
  pub data_dim: usize,
  pub classes: Option<HashSet<usize>>
}

impl<TargetType, DType> Dataset<TargetType, DType>
  where TargetType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Debug,
        DType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Float + Debug {

  /**
   * Initializing Dataset structure by parsing
   * data file from disk.
   * !currently supports only libsvm file format
   */
  pub fn new(path: &Path, task: &LearningTask) -> Self {
    let ext = match path.extension() {
      Some(ext) => ext,
      None => panic!("Wrong file format, supports only libsvm"),
    };

    let data = if ext == "libsvm" {
      let file = match File::open(path) {
        Ok(file) => file,
        Err(error) => {
          panic!("Failed to open file: {:?}. Error: {:?}",
            path, error.description())
        }
      };
      Dataset::<TargetType, DType>::parse_libsvm(&file, task)
    } else {
      panic!("Wrong file format, supports only libsvm");
    };
    let dim: usize = data.0[0].1.len();
    Dataset {
      data: data.0,
      classes: data.1,
      data_dim: dim
    }
  }

}