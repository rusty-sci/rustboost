extern crate clap;
extern crate rustboost;

use clap::{Arg, App};

use rustboost::rustboost::run;
use rustboost::config::Config;
use rustboost::config::{
  DEFAULT_TASK,
  DEFAULT_LT,
  // DEFAULT_OBJECTIVE
};

fn main() {
  let matches = App::new("rustboost")
    .version("0.1.2")
    .author("MoonLight <ixav1@icloud.com>")
    .about("Rust gradient boosting library")
    .arg(Arg::with_name("INPUT")
      .help("Path to data file. Supports only libsvm format.")
      .required(true)
      .index(1))
    .arg(Arg::with_name("TASK")
      .help("Performing task.")
      .takes_value(true)
      .short("t")
      .long("task")
      .default_value(DEFAULT_TASK)
      .possible_values(&["train", "pred", "val"]))
    .arg(Arg::with_name("CONFIG")
      .help("Path to config file, file must be in JSON format.")
      .short("c")
      .long("config")
      .takes_value(true))
    .arg(Arg::with_name("MODEL")
      .help("Model to train. Tree or boosting model.")
      .short("m")
      .long("model")
      .takes_value(true))
    .arg(Arg::with_name("LEARNING_TASK")
      .help("Specify the learning task.")
      .short("l")
      .long("learning_task")
      .default_value(DEFAULT_LT)
      .possible_values(&["regression", "classification"])
      .takes_value(true))
    .arg(Arg::with_name("OBJECTIVE")
      .help("Specify the corresponding learning objective.")
//       .long_help("Specify the learning task and the corresponding
// learning objective, default = classif:tree.")
      .long("objective")
      .short("o")
      .takes_value(true))
      // .default_value(DEFAULT_OBJECTIVE)
      // .possible_values(&["reg:tree", "classif:tree"]))
    .get_matches();

  let path = matches.value_of("INPUT").unwrap();
  let config = Config::default()
    .task(matches.value_of("TASK").unwrap())
    .learning_task(matches.value_of("LEARNING_TASK").unwrap());
  run(path, config);
}