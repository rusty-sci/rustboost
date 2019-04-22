use std::path::Path;

pub const DEFAULT_DTYPE: &'static str = "float32";
pub const DEFAULT_TASK: &'static str = "train";
pub const DEFAULT_OBJECTIVE: &'static str = "classif:tree";

#[derive(Debug)]
pub enum DType {
  Float32,
  Float64
}

impl DType {

  pub fn new(dtype: &str) -> Self {
    match dtype {
      "float32" => DType::Float32,
      "float64" => DType::Float64,
      _ => panic!("Wrong dtype argument.")
    }
  }

}

#[derive(Debug)]
pub enum Task {
  Train,
  Pred,
  Val
}

impl Task {

  pub fn new(dtype: &str) -> Self {
    match dtype {
      "train" => Task::Train,
      "pred" => Task::Pred,
      "val" => Task::Val,
      _ => panic!("Wrong task argument.")
    }
  }

}

#[derive(Debug)]
pub enum Objective {
  Tree
}

impl Objective {

  pub fn new(dtype: &str) -> Self {
    match dtype {
      "tree" => Objective::Tree,
      _ => panic!("Wrong objective argument.")
    }
  }

}

#[derive(Debug)]
pub enum LearningTask {
  Regression,
  Classification
}

impl LearningTask {

  pub fn new(dtype: &str) -> Self {
    match dtype {
      "reg" => LearningTask::Regression,
      "classif" => LearningTask::Classification,
      _ => panic!("Wrong objective argument.")
    }
  }

}

#[derive(Debug)]
pub struct Config<'c> {
  pub path: &'c Path,
  pub dtype: DType,
  pub task: Task,
  pub learning_task: LearningTask,
  pub objective: Objective
}

impl<'c> Config<'c> {

  pub fn default(path: &'c Path) -> Self {
    Config {
      path: path,
      dtype: DType::new(DEFAULT_DTYPE),
      task: Task::new(DEFAULT_TASK),
      learning_task: LearningTask::new(DEFAULT_OBJECTIVE
        .split(":").collect::<Vec<&str>>()[0]),
      objective: Objective::new(DEFAULT_OBJECTIVE
        .split(":").collect::<Vec<&str>>()[1])
    }
  }

  pub fn dtype(self, dtype: &str) -> Self {
    Config {
      dtype: DType::new(dtype),
      ..self
    }
  }

  pub fn task(self, task: &str) -> Self {
    Config {
      task: Task::new(task),
      ..self
    }
  }

  pub fn objective(self, objective: &str) -> Self {
    let objectives = objective.split(":").collect::<Vec<&str>>();
    println!("{}", objectives[0]);
    Config {
      learning_task: LearningTask::new(objectives[0]),
      objective: Objective::new(objectives[1]),
      ..self
    }
  }

}