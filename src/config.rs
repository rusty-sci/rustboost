// use std::path::Path;

// pub const DEFAULT_DTYPE: &'static str = "float32";
pub const DEFAULT_TASK: &'static str = "train";
pub const DEFAULT_LT: &'static str = "regression";
pub const DEFAULT_MODEL: &'static str = "tree";
// pub const DEFAULT_OBJECTIVE: &'static str = "classif:tree";

// #[derive(Debug)]
// pub enum DType {
//   Float32,
//   Float64
// }

// impl DType {

//   pub fn new(dtype: &str) -> Self {
//     match dtype {
//       "float32" => DType::Float32,
//       "float64" => DType::Float64,
//       _ => panic!("Wrong dtype argument.")
//     }
//   }

// }

#[derive(Debug)]
pub enum Task {
  Train,
  Pred,
  Val
}

impl Task {

  pub fn new(task: &str) -> Self {
    match task {
      "train" => Task::Train,
      "pred" => Task::Pred,
      "val" => Task::Val,
      _ => panic!("Wrong task argument.")
    }
  }

}

#[derive(Debug)]
pub enum Model {
  Tree
}

impl Model {

  pub fn new(model: &str) -> Self {
    match model {
      "tree" => Model::Tree,
      _ => panic!("Wrong model argument.")
    }
  }

}

#[derive(Debug)]
pub enum LearningTask {
  Regression,
  Classification
}

impl LearningTask {

  pub fn new(lt: &str) -> Self {
    match lt {
      "regression" => LearningTask::Regression,
      "classification" => LearningTask::Classification,
      _ => panic!("Wrong objective argument.")
    }
  }

}

#[derive(Debug)]
pub struct Config {
  pub task: Task,
  pub learning_task: LearningTask,
  pub model: Model
}

impl Config {

  pub fn default() -> Self {
    Config {
      task: Task::new(DEFAULT_TASK),
      learning_task: LearningTask::new(DEFAULT_LT),
      model: Model::new(DEFAULT_MODEL)
    }
  }

  pub fn task(self, task: &str) -> Self {
    Config {
      task: Task::new(task),
      ..self
    }
  }

  pub fn model(self, model: &str) -> Self {
    Config {
      model: Model::new(model),
      ..self
    }
  }

  pub fn learning_task(self, learning_task: &str) -> Self {
    Config {
      learning_task: LearningTask::new(learning_task),
      ..self
    }
  }

}