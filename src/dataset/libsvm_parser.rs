extern crate num;

use std::fs::File;
use std::io::{BufReader, BufRead};
use std::collections::HashSet;
use std::str::FromStr;
use std::fmt::Debug;

use self::num::{ToPrimitive, Num, FromPrimitive, Float};
use super::{DataType, Dataset};
use super::super::config::{LearningTask};

impl<TargetType, DType> Dataset<TargetType, DType>
  where TargetType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Debug,
        DType: Copy + FromStr + ToPrimitive + Num + FromPrimitive + Float + Debug {

  /**
   * Libsvm parsing function.
   * TODO: Implement out of core support for large data files.
   */
  pub fn parse_libsvm(file: &File, task: &LearningTask) ->
    (DataType<TargetType, DType>, Option<HashSet<usize>>) {

    let mut buf_reader = BufReader::new(file);
    let mut line = String::new();
    let mut data: DataType<TargetType, DType> = Vec::new();
    let mut classes: Option<HashSet<usize>> = None;

    match *task {
      LearningTask::Classification => {
        classes = Some(HashSet::new());
      },
      LearningTask::Regression => ()
    }
    
    while buf_reader.read_line(&mut line)
      .expect("Error in reading file") > 0 {
      line = line.trim().to_string();
      let mut sample: Vec<DType> = Vec::new();
      let mut target: Option<TargetType> = None;
      for (i, value) in line.split_whitespace().enumerate() {
        if i == 0 {
          target = match value.parse::<TargetType>() {
            Ok(target) => Some(target),
            Err(_) => panic!("TODO: use option, class {:?}", value)
          };
          match classes {
            Some(ref mut classes) => {
              let uniq_class = target.unwrap().to_usize().unwrap();
              classes.insert(uniq_class);
            },
            None => ()
          }
        } else {
          let feature: Vec<&str> = value.split(':').collect();
          match feature.last() {
            Some(feature) => {
              let f_value = match feature.parse::<DType>() {
                Ok(val) => val,
                Err(_) => panic!("TODO: use option, feature")
              };
              sample.push(f_value);
            },
            None => panic!("Error in parsing libsm data file")
          }
        }
      }
      data.push((target.unwrap(), sample));
      line.clear();
    }
    (data, classes)
  }

}