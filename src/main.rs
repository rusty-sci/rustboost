extern crate clap;
extern crate rustboost;

use clap::{Arg, App};
use std::path::Path;

use rustboost::rustboost::run;
use rustboost::config::Config;
use rustboost::config::{
  DEFAULT_DTYPE,
  DEFAULT_TASK,
  DEFAULT_OBJECTIVE
};

fn main() {
  let matches = App::new("RustBoost")
    .version("0.1")
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
    .arg(Arg::with_name("DTYPE")
      .help("Data type, can be either float32 or float64.")
      .long("dtype")
      .takes_value(true)
      .default_value(DEFAULT_DTYPE)
      .possible_values(&["float32", "float64"]))
    .arg(Arg::with_name("OBJECTIVE")
      .help("Objective and learning task.")
      .long_help("Specify the learning task and the corresponding
learning objective, default = classif:tree.")
      .long("objective")
      .short("o")
      .takes_value(true)
      .default_value(DEFAULT_OBJECTIVE)
      .possible_values(&["reg:tree", "classif:tree"]))
    .get_matches();

  let path = Path::new(matches.value_of("INPUT").unwrap());
  let config = Config::default(path)
    .dtype(matches.value_of("DTYPE").unwrap())
    .task(matches.value_of("TASK").unwrap())
    .objective(matches.value_of("OBJECTIVE").unwrap());
  run(config);
}