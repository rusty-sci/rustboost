// use std::time::{Duration, Instant};
use std::rc::Rc;

use super::config::*;
use super::dataset::Dataset;
use super::tree::*;

// macro_rules! run_rustboost {
//   ($target_type:ty, $config:expr) => {
//     let path = $config.path;
//     let learning_task = $config.learning_task;
//     let dataset = Rc::new(RefCell::new(Dataset::<$target_type, f32>::new(path, &learning_task)));
//     // println!("{:?}", dataset.borrow().data);
//     let mut tree = Tree::new(Rc::clone(&dataset))
//       .learning_task(learning_task)
//       .min_node_samples(10)
//       .max_tree_depth(2);
//     let start = Instant::now();
//     tree.fit();
//     let duration = start.elapsed();

//     println!("Time elapsed in tree.fit() is: {:?}", duration);
//     // println!("{:#?}", dataset.borrow().data);
//   }
// }

pub fn run(path: &str, config: Config) {
  env_logger::init();
  debug!("{:#?}", config);
  let mut dataset;
  match config.learning_task {
    LearningTask::Regression => {
      dataset = Dataset::from_libsvm(path, false);
    },
    LearningTask::Classification => {
      dataset = Dataset::from_libsvm(path, true);
    }
  }
  let dataset = Rc::new(dataset);
  // debug!("{:#?}", dataset);
  let mut tree = Tree::new(Rc::clone(&dataset))
    .learning_task(config.learning_task)
    .min_node_samples(10)
    .max_tree_depth(2);
  debug!("{:#?}", tree);
  debug!("Fitting!");
  tree.fit();
  debug!("{:#?}", tree);
}